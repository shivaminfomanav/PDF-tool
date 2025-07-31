import streamlit as st
import tempfile
import os
import pandas as pd
import re
from typing import List, Dict, Any, Tuple
import requests
from urllib.parse import urlparse
import hashlib

# PDF extraction tools
import pdfplumber
import fitz  # PyMuPDF
import camelot
import tabula
from pdf2image import convert_from_path
import pytesseract

st.set_page_config("Fixed SEC Table Extractor", layout="wide")
st.title(" Fixed SEC Table Extractor - Column Names & Values")

# Session state
if 'results' not in st.session_state:
    st.session_state.results = None

def download_pdf_from_url(url: str) -> str:
    """Download PDF from URL"""
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        response = requests.get(url, headers=headers, timeout=30, stream=True)
        response.raise_for_status()
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            for chunk in response.iter_content(chunk_size=8192):
                tmp.write(chunk)
            return tmp.name
    except Exception as e:
        raise Exception(f"Failed to download PDF: {str(e)}")

def create_content_hash(table: List[List]) -> str:
    """Create unique hash based on table content"""
    if not table or not table[0]:
        return "empty"
    
    # Use first 2 rows and first 2 columns for unique signature
    content_sample = []
    for i, row in enumerate(table[:2]):
        row_sample = row[:2] if len(row) >= 2 else row
        content_sample.extend(row_sample)
    
    content_str = ''.join(str(cell).strip().lower() for cell in content_sample if cell)
    return hashlib.md5(content_str.encode()).hexdigest()[:8]

def smart_clean_field(field_name: str) -> str:
    """Smart field cleaning that preserves important distinctions"""
    if not field_name or not str(field_name).strip():
        return ""
    
    cleaned = str(field_name).strip()
    
    # Remove excessive punctuation but keep meaningful separators
    cleaned = re.sub(r'[^\w\s\-()&/.,:]', '', cleaned)
    
    # Standardize multiple spaces to single space
    cleaned = re.sub(r'\s+', ' ', cleaned)
    
    # Remove leading/trailing punctuation but keep internal
    cleaned = cleaned.strip('-.,()')
    
    return cleaned

def extract_tables_multi_method(pdf_path: str) -> List[Dict]:
    """Extract tables using multiple methods with improved deduplication"""
    all_tables = []
    
    # Method 1: PDFPlumber
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                tables = page.extract_tables()
                for table in tables:
                    if table and len(table) > 1 and len(table[0]) > 1:
                        all_tables.append({
                            'page': page_num + 1,
                            'table': table,
                            'method': 'pdfplumber',
                            'content_hash': create_content_hash(table)
                        })
    except Exception as e:
        st.warning(f"PDFPlumber error: {e}")
    
    # Method 2: PyMuPDF
    try:
        doc = fitz.open(pdf_path)
        for page_num in range(doc.page_count):
            page = doc[page_num]
            tables = page.find_tables()
            for table in tables:
                extracted = table.extract()
                if extracted and len(extracted) > 1:
                    all_tables.append({
                        'page': page_num + 1,
                        'table': extracted,
                        'method': 'pymupdf',
                        'content_hash': create_content_hash(extracted)
                    })
        doc.close()
    except Exception as e:
        st.warning(f"PyMuPDF error: {e}")
    
    # Method 3: Camelot
    try:
        tables = camelot.read_pdf(pdf_path, pages='all', flavor='lattice')
        for table in tables:
            if not table.df.empty and len(table.df) > 1:
                table_data = table.df.values.tolist()
                all_tables.append({
                    'page': table.page,
                    'table': table_data,
                    'method': 'camelot',
                    'content_hash': create_content_hash(table_data)
                })
    except Exception as e:
        st.warning(f"Camelot error: {e}")
    
    # Method 4: Tabula
    try:
        tables = tabula.read_pdf(pdf_path, pages='all', multiple_tables=True)
        for i, df in enumerate(tables):
            if not df.empty and len(df) > 1:
                table_data = df.values.tolist()
                all_tables.append({
                    'page': i + 1,
                    'table': table_data,
                    'method': 'tabula',
                    'content_hash': create_content_hash(table_data)
                })
    except Exception as e:
        st.warning(f"Tabula error: {e}")
    
    return all_tables

def deduplicate_tables(tables: List[Dict]) -> List[Dict]:
    """Remove duplicate tables based on content hash"""
    seen_hashes = set()
    unique_tables = []
    
    # Prioritize methods: camelot > tabula > pdfplumber > pymupdf
    method_priority = {'camelot': 1, 'tabula': 2, 'pdfplumber': 3, 'pymupdf': 4}
    
    # Sort by priority
    tables_sorted = sorted(tables, key=lambda x: method_priority.get(x['method'], 5))
    
    for table in tables_sorted:
        content_hash = table['content_hash']
        if content_hash not in seen_hashes:
            seen_hashes.add(content_hash)
            unique_tables.append(table)
    
    return unique_tables

def extract_financial_keywords():
    """Financial keywords for classification"""
    return [
        'revenue', 'sales', 'income', 'earnings', 'profit', 'loss',
        'assets', 'liabilities', 'equity', 'cash', 'debt',
        'expenses', 'costs', 'operating', 'net', 'total', 'gross',
        'current', 'receivables', 'payables', 'inventory',
        'property', 'equipment', 'goodwill', 'retained'
    ]

def is_financial_field(field_name: str) -> bool:
    """Check if field contains financial keywords"""
    if not field_name:
        return False
    
    field_lower = field_name.lower()
    keywords = extract_financial_keywords()
    
    return any(keyword in field_lower for keyword in keywords)

def extract_numbers_from_cells(cells: List[str]) -> List[float]:
    """Extract numbers from table cells with SEC format handling"""
    numbers = []
    
    for cell in cells:
        if not cell or not str(cell).strip():
            continue
        
        cell_str = str(cell).strip()
        
        # Handle parentheses as negative (SEC standard)
        is_negative = cell_str.startswith('(') and cell_str.endswith(')')
        if is_negative:
            cell_str = cell_str[1:-1]
        
        # Remove currency symbols and commas
        cell_str = re.sub(r'[$€£¥₹,\s]', '', cell_str)
        
        # Handle dashes as zero
        if cell_str in ['-', '—', '–', '']:
            numbers.append(0.0)
            continue
        
        # Extract number
        try:
            # Try direct conversion first
            number = float(cell_str)
            if is_negative:
                number = -number
            numbers.append(number)
        except ValueError:
            # Try to find first number in the string
            number_match = re.search(r'-?\d+\.?\d*', cell_str)
            if number_match:
                try:
                    number = float(number_match.group())
                    if is_negative and number > 0:
                        number = -number
                    numbers.append(number)
                except ValueError:
                    continue
    
    return numbers

def process_tables_to_columns(tables: List[Dict], source_name: str) -> pd.DataFrame:
    """Process tables and extract column names with values"""
    results = []
    field_counter = {}  # Track duplicate field names
    
    for table_info in tables:
        table = table_info['table']
        page = table_info['page']
        method = table_info['method']
        
        if not table or len(table) < 2:
            continue
        
        # Process each row
        for row_idx, row in enumerate(table):
            if not row or len(row) < 2:
                continue
            
            # First column as field name
            raw_field = str(row[0]).strip()
            if not raw_field or len(raw_field) < 2:
                continue
            
            # Smart cleaning that preserves distinctions
            clean_field = smart_clean_field(raw_field)
            if not clean_field or len(clean_field) < 2:
                continue
            
            # Skip non-financial fields unless they contain numbers
            value_cells = [str(cell) for cell in row[1:] if cell]
            has_numbers = any(re.search(r'\d', str(cell)) for cell in value_cells)
            
            if not is_financial_field(clean_field) and not has_numbers:
                continue
            
            # Handle duplicate field names by making them unique
            if clean_field in field_counter:
                field_counter[clean_field] += 1
                unique_field = f"{clean_field} (P{page}-{field_counter[clean_field]})"
            else:
                field_counter[clean_field] = 0
                unique_field = clean_field
            
            # Extract numbers from value columns
            numbers = extract_numbers_from_cells(value_cells)
            
            if numbers:  # Only include rows with valid numbers
                # Create result row with column structure
                result_row = {
                    'Source': source_name,
                    'Page': page,
                    'Method': method,
                    'Field_Name': unique_field,
                    'Raw_Field': raw_field,
                    'Values': numbers,
                    'Total': sum(numbers),
                    'Count': len(numbers)
                }
                
                # Add individual value columns
                for i, value in enumerate(numbers[:5]):  # Limit to first 5 values
                    result_row[f'Value_{i+1}'] = value
                
                results.append(result_row)
    
    if not results:
        return pd.DataFrame()
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Fill NaN values in Value columns with empty string for display
    value_cols = [col for col in df.columns if col.startswith('Value_')]
    for col in value_cols:
        df[col] = df[col].fillna('')
    
    return df

def validate_results(df: pd.DataFrame) -> Dict[str, Any]:
    """Validate extraction results"""
    if df.empty:
        return {'status': 'No data extracted', 'issues': ['Empty result']}
    
    validation = {
        'total_rows': len(df),
        'unique_fields': df['Field_Name'].nunique(),
        'pages_covered': df['Page'].nunique(),
        'methods_used': df['Method'].nunique(),
        'duplicate_ratio': 1 - (df['Field_Name'].nunique() / len(df)),
        'issues': []
    }
    
    # Check for issues
    if validation['duplicate_ratio'] > 0.3:
        validation['issues'].append("High field duplication detected")
    
    if validation['unique_fields'] < 5:
        validation['issues'].append("Very few unique fields extracted")
    
    validation['status'] = 'Good' if not validation['issues'] else 'Issues detected'
    
    return validation

# Streamlit Interface
st.markdown("##  Input Options")

# Create tabs
tab1, tab2, tab3 = st.tabs([" Upload File", " URL", " Multiple Files"])

with tab1:
    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

with tab2:
    pdf_url = st.text_input("PDF URL:", placeholder="https://example.com/file.pdf")

with tab3:
    uploaded_files = st.file_uploader("Upload Multiple PDFs", type=["pdf"], accept_multiple_files=True)

# Configuration
st.sidebar.header("⚙️ Settings")
show_raw_fields = st.sidebar.checkbox("Show Original Field Names", value=False)
show_metadata = st.sidebar.checkbox("Show Page & Method Info", value=True)
min_value_threshold = st.sidebar.slider("Minimum Value Count", 1, 5, 1)

# Process single file
if uploaded_file and st.button(" Extract Tables"):
    with st.spinner("Extracting tables..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            temp_path = tmp.name
        
        try:
            # Extract tables
            raw_tables = extract_tables_multi_method(temp_path)
            unique_tables = deduplicate_tables(raw_tables)
            
            # Process to column format
            df = process_tables_to_columns(unique_tables, uploaded_file.name)
            
            if not df.empty:
                # Filter by minimum value count
                df = df[df['Count'] >= min_value_threshold]
                
                validation = validate_results(df)
                st.session_state.results = df
                
                # Show summary
                col1, col2, col3, col4 = st.columns(4)
                col1.metric(" Fields", validation['total_rows'])
                col2.metric(" Unique", validation['unique_fields'])
                col3.metric(" Pages", validation['pages_covered'])
                col4.metric(" Methods", validation['methods_used'])
                
                if validation['issues']:
                    st.warning(f" Issues: {', '.join(validation['issues'])}")
                else:
                    st.success(" Extraction completed successfully!")
                
                # Display results
                st.markdown("##  Extracted Data")
                
                # Prepare display DataFrame
                display_df = df.copy()
                
                # Select columns to show
                base_cols = ['Field_Name']
                if show_raw_fields:
                    base_cols.append('Raw_Field')
                if show_metadata:
                    base_cols.extend(['Page', 'Method'])
                
                # Add value columns
                value_cols = [col for col in df.columns if col.startswith('Value_') and not df[col].isna().all()]
                display_cols = base_cols + value_cols + ['Total']
                
                # Format numeric columns
                for col in value_cols + ['Total']:
                    display_df[col] = display_df[col].apply(
                        lambda x: f"${x:,.0f}" if pd.notna(x) and x != '' and x != 0 else ''
                    )
                
                st.dataframe(display_df[display_cols], use_container_width=True, height=400)
                
                # Download button
                csv_data = df.to_csv(index=False)
                st.download_button(
                    " Download CSV",
                    csv_data,
                    f"{uploaded_file.name}_extracted.csv",
                    mime="text/csv"
                )
            else:
                st.error("❌ No financial data extracted")
        
        finally:
            os.unlink(temp_path)

# Process URL
if pdf_url and st.button(" Extract from URL"):
    try:
        with st.spinner("Downloading PDF..."):
            temp_path = download_pdf_from_url(pdf_url)
        
        with st.spinner("Extracting tables..."):
            raw_tables = extract_tables_multi_method(temp_path)
            unique_tables = deduplicate_tables(raw_tables)
            df = process_tables_to_columns(unique_tables, f"URL: {pdf_url}")
            
            if not df.empty:
                df = df[df['Count'] >= min_value_threshold]
                validation = validate_results(df)
                st.session_state.results = df
                
                st.success(f" Extracted {len(df)} fields from URL!")
                
                # Display results (same as above)
                display_df = df.copy()
                base_cols = ['Field_Name']
                if show_raw_fields:
                    base_cols.append('Raw_Field')
                if show_metadata:
                    base_cols.extend(['Page', 'Method'])
                
                value_cols = [col for col in df.columns if col.startswith('Value_') and not df[col].isna().all()]
                display_cols = base_cols + value_cols + ['Total']
                
                for col in value_cols + ['Total']:
                    display_df[col] = display_df[col].apply(
                        lambda x: f"${x:,.0f}" if pd.notna(x) and x != '' and x != 0 else ''
                    )
                
                st.dataframe(display_df[display_cols], use_container_width=True)
                
                st.download_button(
                    " Download CSV",
                    df.to_csv(index=False),
                    "url_extracted.csv",
                    mime="text/csv"
                )
            else:
                st.error(" No financial data extracted from URL")
    
    except Exception as e:
        st.error(f" Error: {e}")
    finally:
        if 'temp_path' in locals():
            os.unlink(temp_path)

# Process multiple files
if uploaded_files and st.button(" Extract from All Files"):
    all_results = []
    
    progress_bar = st.progress(0)
    for i, file in enumerate(uploaded_files):
        st.write(f"Processing {file.name}...")
        progress_bar.progress((i + 1) / len(uploaded_files))
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file.read())
            temp_path = tmp.name
        
        try:
            raw_tables = extract_tables_multi_method(temp_path)
            unique_tables = deduplicate_tables(raw_tables)
            df = process_tables_to_columns(unique_tables, file.name)
            
            if not df.empty:
                df = df[df['Count'] >= min_value_threshold]
                all_results.append(df)
        
        except Exception as e:
            st.warning(f"Error processing {file.name}: {e}")
        finally:
            os.unlink(temp_path)
    
    if all_results:
        combined_df = pd.concat(all_results, ignore_index=True)
        st.session_state.results = combined_df
        
        st.success(f" Batch processing complete! {len(combined_df)} total fields extracted")
        
        # Summary by file
        summary = combined_df.groupby('Source').agg({
            'Field_Name': 'count',
            'Total': 'sum'
        }).rename(columns={'Field_Name': 'Fields_Count', 'Total': 'Total_Value'})
        
        st.markdown("##  Summary by File")
        st.dataframe(summary, use_container_width=True)
        
        # Display all results
        st.markdown("##  All Extracted Data")
        display_df = combined_df.copy()
        
        base_cols = ['Source', 'Field_Name']
        if show_metadata:
            base_cols.extend(['Page', 'Method'])
        
        value_cols = [col for col in combined_df.columns if col.startswith('Value_') and not combined_df[col].isna().all()]
        display_cols = base_cols + value_cols + ['Total']
        
        for col in value_cols + ['Total']:
            display_df[col] = display_df[col].apply(
                lambda x: f"${x:,.0f}" if pd.notna(x) and x != '' and x != 0 else ''
            )
        
        st.dataframe(display_df[display_cols], use_container_width=True, height=500)
        
        st.download_button(
            " Download Batch Results",
            combined_df.to_csv(index=False),
            "batch_extracted.csv",
            mime="text/csv"
        )
    else:
        st.error("❌ No data extracted from any files")

# Info section
if not uploaded_file and not pdf_url and not uploaded_files:
    st.info("📤 Choose an input method above to start extraction")
    
    st.markdown("""
    ###  Key Improvements:
    - **Fixed identical output issue** - Smart field cleaning preserves distinctions
    - **Content-based deduplication** - Tables identified by actual content, not just size  
    - **Column-focused extraction** - Shows field names with their corresponding values
    - **Enhanced validation** - Detects extraction quality issues
    - **Multiple extraction methods** - PDFPlumber, PyMuPDF, Camelot, Tabula
    
    ###  Output Format:
    - **Field_Name**: Cleaned field name (preserves years, categories)
    - **Value_1, Value_2, etc.**: Individual column values
    - **Total**: Sum of all values in the row
    - **Source/Page/Method**: Metadata for tracking
    """)
