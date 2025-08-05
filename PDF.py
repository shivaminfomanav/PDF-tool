import streamlit as st
import pdfplumber
import re
import joblib
import os
from datetime import datetime
from collections import defaultdict, Counter
import json
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')


class EnhancedFinancialLineClassifier:
    """
    Advanced ensemble classifier with multiple ML models and hyperparameter optimization
    for determining HEADER vs DATA_ROW classification in financial documents.
    """
    
    def __init__(self, model_type: str = 'ensemble'):
        self.model_type = model_type
        self.model_path = f"enhanced_classifier_{self.model_type}.joblib"
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 3), max_features=5000, stop_words='english')
        self.accuracy_score = 0.0
        self.classification_report = ""
        
        if os.path.exists(self.model_path):
            self.model, self.vectorizer, self.accuracy_score = self.load_model()
        else:
            self.model = None

    def _create_advanced_features(self, lines: List[Dict]):
        """Enhanced feature engineering with more sophisticated text and layout features."""
        raw_texts = [line['text'] for line in lines]
        self.vectorizer.fit(raw_texts)
        text_features = self.vectorizer.transform(raw_texts).toarray()
        
        advanced_features = []
        for line in lines:
            text = line['text']
            # Enhanced feature set for better classification
            features = [
                line.get('x0', 0),  # Indentation
                len(text),  # Text length
                len(text.split()),  # Word count
                sum(c.isdigit() for c in text),  # Digit count
                sum(c.isupper() for c in text),  # Uppercase count
                text.count('$'),  # Dollar sign count
                text.count('%'),  # Percentage count
                text.count('('),  # Opening parenthesis count
                1 if re.search(r'\(\s*\d', text) else 0,  # Negative number pattern
                1 if text.endswith(':') else 0,  # Header indicator
                1 if text.isupper() else 0,  # All caps indicator
                1 if re.search(r'^\d+\.', text) else 0,  # Numbered list indicator
                len(re.findall(r'[,.]', text)),  # Punctuation count
                1 if re.search(r'\d{4}', text) else 0,  # Year pattern
                text.count('\t'),  # Tab count for indentation
                len([w for w in text.split() if w.isalpha()]),  # Alphabetic word count
            ]
            advanced_features.append(features)
        
        return np.hstack((text_features, np.array(advanced_features)))

    def _create_ensemble_model(self):
        """Creates an ensemble model combining multiple classifiers with optimized hyperparameters."""
        # Optimized individual models with hyperparameter tuning
        rf_model = RandomForestClassifier(
            n_estimators=200, max_depth=15, min_samples_split=5,
            min_samples_leaf=2, random_state=42, class_weight='balanced'
        )
        
        gb_model = GradientBoostingClassifier(
            n_estimators=150, learning_rate=0.1, max_depth=6,
            min_samples_split=4, random_state=42
        )
        
        lr_model = LogisticRegression(
            random_state=42, class_weight='balanced', max_iter=2000,
            C=1.0, solver='liblinear'
        )
        
        ada_model = AdaBoostClassifier(
            n_estimators=100, learning_rate=0.8, random_state=42
        )
        
        et_model = ExtraTreesClassifier(
            n_estimators=150, max_depth=12, min_samples_split=3,
            random_state=42, class_weight='balanced'
        )
        
        # Voting ensemble combining all models
        ensemble_model = VotingClassifier(
            estimators=[
                ('rf', rf_model),
                ('gb', gb_model), 
                ('lr', lr_model),
                ('ada', ada_model),
                ('et', et_model)
            ],
            voting='soft'
        )
        
        return ensemble_model

    def train_with_optimization(self, lines: List[Dict], labels: List[str]):
        """Train model with hyperparameter optimization and cross-validation with progress tracking."""
        # Initialize progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Step 1: Feature Engineering
        status_text.text("🔧 Creating advanced features...")
        progress_bar.progress(10)
        
        features = self._create_advanced_features(lines)
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        progress_bar.progress(20)
        status_text.text("📊 Data split completed...")
        
        print(f"Training enhanced '{self.model_type}' classification model...")
        
        # Step 2: Model Selection and Creation
        status_text.text(f"🏗️ Building {self.model_type} model...")
        progress_bar.progress(30)
        
        if self.model_type == 'ensemble':
            self.model = self._create_ensemble_model()
        elif self.model_type == 'optimized_rf':
            # Hyperparameter optimization for Random Forest
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [10, 15, 20],
                'min_samples_split': [2, 5]
            }
            rf = RandomForestClassifier(random_state=42, class_weight='balanced')
            self.model = GridSearchCV(rf, param_grid, cv=3, scoring='accuracy')
        else:
            # Default optimized models
            if self.model_type == 'random_forest':
                self.model = RandomForestClassifier(
                    n_estimators=200, max_depth=15, random_state=42, class_weight='balanced'
                )
            elif self.model_type == 'gradient_boosting':
                self.model = GradientBoostingClassifier(
                    n_estimators=150, learning_rate=0.1, max_depth=6, random_state=42
                )
        
        progress_bar.progress(50)
        
        # Step 3: Model Training
        status_text.text("🎯 Training model...")
        self.model.fit(X_train, y_train)
        progress_bar.progress(70)
        
        # Step 4: Model Evaluation
        status_text.text("📈 Evaluating model performance...")
        y_pred = self.model.predict(X_test)
        self.accuracy_score = accuracy_score(y_test, y_pred)
        self.classification_report = classification_report(y_test, y_pred)
        progress_bar.progress(85)
        
        # Step 5: Cross-validation
        status_text.text("🔄 Running cross-validation...")
        cv_scores = cross_val_score(self.model, features, labels, cv=5)
        print(f"Cross-validation accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        progress_bar.progress(95)
        
        # Step 6: Save Model
        status_text.text("💾 Saving trained model...")
        self.save_model()
        progress_bar.progress(100)
        
        status_text.text(f"✅ Model training complete! Accuracy: {self.accuracy_score:.3f}")
        print(f"Model training complete. Test accuracy: {self.accuracy_score:.3f}")

    def predict(self, line: Dict):
        """Enhanced prediction with confidence scoring."""
        if not self.model:
            raise RuntimeError("Model not trained.")
        
        text_features = self.vectorizer.transform([line['text']]).toarray()
        
        # Enhanced feature extraction for single prediction
        text = line['text']
        other_features = [
            line.get('x0', 0), len(text), len(text.split()),
            sum(c.isdigit() for c in text), sum(c.isupper() for c in text),
            text.count('$'), text.count('%'), text.count('('),
            1 if re.search(r'\(\s*\d', text) else 0,
            1 if text.endswith(':') else 0,
            1 if text.isupper() else 0,
            1 if re.search(r'^\d+\.', text) else 0,
            len(re.findall(r'[,.]', text)),
            1 if re.search(r'\d{4}', text) else 0,
            text.count('\t'),
            len([w for w in text.split() if w.isalpha()])
        ]
        
        full_features = np.hstack((text_features, np.array([other_features])))
        return self.model.predict(full_features)[0]

    def save_model(self):
        """Save model with accuracy metrics."""
        joblib.dump((self.model, self.vectorizer, self.accuracy_score), self.model_path)

    def load_model(self):
        """Load pre-trained model with metrics."""
        print(f"Loading pre-trained '{self.model_type}' model...")
        return joblib.load(self.model_path)


class CleanPDFExtractor:
    """
    Clean PDF extraction using both PDFPlumber text extraction and tabular extraction.
    """
    
    def __init__(self):
        pass
    
    def extract_with_pdfplumber(self, pdf_file) -> List[Dict]:
        """Extract using PDFPlumber for text-based extraction with progress tracking."""
        lines = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            status_text.text("📄 Opening PDF with PDFPlumber...")
            with pdfplumber.open(pdf_file) as pdf:
                total_pages = len(pdf.pages)
                
                for page_num, page in enumerate(pdf.pages):
                    status_text.text(f"📄 PDFPlumber: Processing page {page_num + 1}/{total_pages}...")
                    progress = int(((page_num + 1) / total_pages) * 50)  # 50% of total progress
                    progress_bar.progress(progress)
                    
                    page_lines = page.extract_text_lines(layout=True)
                    for line in page_lines:
                        line['page'] = page_num + 1
                        line['method'] = 'pdfplumber'
                        line['extraction_order'] = len(lines)  # Track extraction order
                        lines.append(line)
                
                status_text.text(f"✅ PDFPlumber extraction complete: {len(lines)} lines")
                progress_bar.progress(50)
                
        except Exception as e:
            st.warning(f"PDFPlumber extraction failed: {str(e)}")
        
        return lines
    
    def extract_with_tabular(self, pdf_file) -> List[Dict]:
        """Extract using PDFPlumber for tabular data extraction with progress tracking."""
        lines = []
        progress_bar = st.progress(50)  # Start from 50%
        status_text = st.empty()
        
        try:
            status_text.text("📊 Opening PDF for tabular extraction...")
            with pdfplumber.open(pdf_file) as pdf:
                total_pages = len(pdf.pages)
                
                for page_num, page in enumerate(pdf.pages):
                    status_text.text(f"📊 Tabular: Processing page {page_num + 1}/{total_pages}...")
                    progress = 50 + int(((page_num + 1) / total_pages) * 50)  # 50-100% of total progress
                    progress_bar.progress(progress)
                    
                    tables = page.extract_tables()
                    for table in tables:
                        for row in table:
                            if row:  # Skip empty rows
                                # Join row cells into single text line
                                row_text = ' '.join([str(cell) if cell else '' for cell in row]).strip()
                                if row_text:
                                    lines.append({
                                        'text': row_text,
                                        'x0': 0,  # Default positioning for tabular data
                                        'page': page_num + 1,
                                        'method': 'tabular',
                                        'extraction_order': len(lines)  # Track extraction order
                                    })
                
                status_text.text(f"✅ Tabular extraction complete: {len(lines)} lines")
                progress_bar.progress(100)
                
        except Exception as e:
            st.warning(f"Tabular extraction failed: {str(e)}")
        
        return lines


class CleanFinancialExtractor:
    """
    Clean financial data extractor with both PDFPlumber and Tabular extraction methods.
    Allows duplicate fields and categories but prevents duplicate values.
    """
    
    def __init__(self, model_type: str):
        self.classifier = EnhancedFinancialLineClassifier(model_type=model_type)
        self.extractor = CleanPDFExtractor()
        
    def _parse_financial_values(self, line_text: str) -> Optional[Dict]:
        """Enhanced parsing with clean value extraction - no commas in individual values."""
        # More precise regex patterns for different value types
        value_patterns = [
            r'\$\s*[\d,]+\.?\d{0,2}',  # Dollar amounts like $1,234.56
            r'\(\s*\$?\s*[\d,]+\.?\d{0,2}\s*\)',  # Negative amounts in parentheses
            r'[\d,]+\.\d{2}(?!\d)',  # Decimal numbers like 1,234.56
            r'[\d,]+\s*%',  # Percentages like 25%
            r'(?<!\d)[\d,]{4,}(?!\d)',  # Large numbers without decimals
        ]
        
        # Find all financial values in the line
        found_values = []
        for pattern in value_patterns:
            matches = re.finditer(pattern, line_text)
            for match in matches:
                found_values.append({
                    'value': match.group().strip(),
                    'position': match.start()
                })
        
        if not found_values:
            return None
        
        # Sort by position to maintain order
        found_values.sort(key=lambda x: x['position'])
        
        # Find the description (text before first financial value)
        first_value_pos = found_values[0]['position']
        description = line_text[:first_value_pos].strip()
        
        # Clean description - remove trailing dots, bullets, etc.
        description = re.sub(r'[\.·•…\-_]+$', '', description).strip()
        
        # Validate description
        if not re.search('[a-zA-Z]{2,}', description) or len(description) < 2:
            return None
        
        # Extract clean values (limit to 4 as requested)
        clean_values = []
        for val_info in found_values[:4]:
            clean_val = val_info['value'].strip()
            # Keep the value as-is, don't remove internal commas
            clean_values.append(clean_val)
        
        return {
            'description': description.title(),
            'values': clean_values
        }

    def _enhanced_bootstrap_training(self, lines: List[Dict]) -> Tuple[List[Dict], List[str]]:
        """Enhanced training data generation with better heuristics."""
        training_lines, labels = [], []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        total_lines = len(lines)
        
        for i, line in enumerate(lines):
            if i % 100 == 0:  # Update progress every 100 lines
                progress = int((i / total_lines) * 100)
                progress_bar.progress(progress)
                status_text.text(f"🤖 Analyzing training data: {i}/{total_lines} lines...")
            
            text = line['text'].strip()
            if not text or len(text) < 3:
                continue
                
            # Enhanced heuristics for classification
            parsed_data = self._parse_financial_values(text)
            
            # DATA_ROW indicators - lines with financial values
            if (parsed_data or 
                re.search(r'\$[\d,]+', text) or 
                re.search(r'\d+\.\d{2}', text) or
                re.search(r'\(\s*[\d,]+', text)):
                labels.append("DATA_ROW")
                training_lines.append(line)
                
            # HEADER indicators - category/section headers
            elif (text.endswith(':') or
                  text.isupper() and len(text.split()) < 6 or
                  re.search(r'^[A-Z][a-z\s]+$', text) and len(text.split()) < 8 or
                  line.get('x0', 0) < 50):  # Left-aligned headers
                labels.append("HEADER")
                training_lines.append(line)
        
        progress_bar.progress(100)
        status_text.text(f"✅ Training data prepared: {len(training_lines)} samples")
        
        return training_lines, labels

    def extract_clean_data(self, pdf_file) -> Dict:
        """Clean extraction allowing duplicate fields/categories but preventing duplicate values."""
        # Initialize main progress tracking
        main_progress = st.progress(0)
        main_status = st.empty()
        
        main_status.text("🚀 Starting PDF extraction process...")
        main_progress.progress(5)
        
        # Extract using PDFPlumber first, then Tabular
        main_status.text("📄 Phase 1: PDFPlumber extraction...")
        pdfplumber_lines = self.extractor.extract_with_pdfplumber(pdf_file)
        main_progress.progress(25)
        
        main_status.text("📊 Phase 2: Tabular extraction...")
        tabular_lines = self.extractor.extract_with_tabular(pdf_file)
        main_progress.progress(50)
        
        # Combine both extraction methods - PDFPlumber first, then Tabular
        all_lines = pdfplumber_lines + tabular_lines
        main_progress.progress(55)
        
        if not all_lines:
            st.warning("No data could be extracted from the document.")
            return {}
        
        main_status.text(f"📊 Combined extraction: {len(all_lines)} total lines")
        
        # Train classifier if needed
        if not self.classifier.model:
            main_status.text("🤖 Phase 3: Training classification model...")
            main_progress.progress(60)
            
            training_lines, labels = self._enhanced_bootstrap_training(all_lines)
            if not training_lines:
                st.warning("Could not generate training data from the document.")
                return {}
            
            main_progress.progress(70)
            self.classifier.train_with_optimization(training_lines, labels)
            main_progress.progress(85)
        
        # Extract data with line-by-line processing
        main_status.text("🔍 Phase 4: Extracting financial data...")
        extracted_data = defaultdict(list)
        header_stack = [(0, "Financial Summary")]
        current_table_items = 0
        
        # Sort all lines by page and extraction order for line-by-line processing
        sorted_lines = sorted(all_lines, key=lambda x: (x.get('page', 1), x.get('extraction_order', 0)))
        total_sorted = len(sorted_lines)
        
        # Process lines in extraction order (line by line)
        for idx, line in enumerate(sorted_lines):
            if idx % 50 == 0:  # Update progress every 50 lines
                progress = 85 + int(((idx + 1) / total_sorted) * 10)
                main_progress.progress(progress)
                main_status.text(f"🔍 Processing line {idx + 1}/{total_sorted}...")
            
            line_text = line['text'].strip()
            indent = line.get('x0', 0)
            method = line.get('method', 'unknown')
            
            if not line_text:
                continue
            
            # Adjust header stack based on indentation
            while indent < header_stack[-1][0] and len(header_stack) > 1:
                header_stack.pop()
            
            # Classify line type
            line_type = self.classifier.predict({'text': line_text, 'x0': indent})
            
            if line_type == "DATA_ROW":
                parsed_data = self._parse_financial_values(line_text)
                if parsed_data:
                    item_name = parsed_data['description']
                    values = parsed_data['values']
                    
                    # Determine parent category
                    parent_category = header_stack[-1][1]
                    
                    # Prepare individual value columns (up to 4 values)
                    value_1 = values[0] if len(values) > 0 else ''
                    value_2 = values[1] if len(values) > 1 else ''
                    value_3 = values[2] if len(values) > 2 else ''
                    value_4 = values[3] if len(values) > 3 else ''
                    
                    extracted_data[parent_category].append({
                        'item': item_name,
                        'value_1': value_1,
                        'value_2': value_2,
                        'value_3': value_3,
                        'value_4': value_4,
                        'page': line.get('page', 1),
                        'method': method,
                        'extraction_order': line.get('extraction_order', 0)
                    })
                    
                    current_table_items += 1
                    
            elif line_type == "HEADER":
                new_header = line_text.title()
                header_stack.append((indent, new_header))
                # Don't prevent duplicate categories - allow them
                if new_header not in extracted_data:
                    extracted_data[new_header] = []
        
        main_progress.progress(95)
        main_status.text("🧹 Phase 5: Cleaning and finalizing data...")
        
        # Modified duplicate handling: Allow duplicate fields and categories, prevent only duplicate values
        final_data = {}
        for category, items in extracted_data.items():
            if items:
                # Only prevent duplicate VALUE COMBINATIONS, allow duplicate field names and categories
                seen_values = set()
                unique_items = []
                for item in items:
                    # Create signature based only on VALUES (not item names or categories)
                    value_signature = f"{item['value_1']}_{item['value_2']}_{item['value_3']}_{item['value_4']}"
                    
                    # Only skip if exact same value combination exists
                    if value_signature not in seen_values or value_signature == "___":  # Allow empty values
                        seen_values.add(value_signature)
                        unique_items.append(item)
                
                # Sort by extraction order to maintain line-by-line processing
                unique_items.sort(key=lambda x: x.get('extraction_order', 0))
                final_data[category] = unique_items
        
        main_progress.progress(100)
        main_status.text(f"✅ Extraction complete! Found {len(final_data)} categories with {sum(len(items) for items in final_data.values())} total items")
        
        return final_data

    def generate_clean_output(self, categorized_data: Dict) -> Tuple[str, Dict, str]:
        """Generate clean output with separate tables for PDFPlumber and Tabular methods."""
        if not categorized_data:
            return "No valid financial data could be extracted.", {}, "Model accuracy: N/A"
        
        total_items = sum(len(items) for items in categorized_data.values())
        
        # Count items by method
        method_counts = defaultdict(int)
        for items in categorized_data.values():
            for item in items:
                method_counts[item.get('method', 'unknown')] += 1
        
        # Model performance report
        accuracy_report = f"""
## Model Performance Report
- **Model Type**: {self.classifier.model_type.replace('_', ' ').title()}
- **Accuracy Score**: {self.classifier.accuracy_score:.3f}
- **Total Items Extracted**: {total_items}
- **Categories Detected**: {len(categorized_data)}
- **PDFPlumber Items**: {method_counts.get('pdfplumber', 0)}
- **Tabular Items**: {method_counts.get('tabular', 0)}
- **Duplicate Policy**: Allow duplicate fields/categories, prevent duplicate values only

### Classification Report
{self.classifier.classification_report}
"""
        
        # Main extraction report header
        markdown = f"# Financial Data Extraction Report (Separate Tables by Method)\n\n"
        markdown += f"**Extraction Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        markdown += f"**Total Items**: {total_items}\n"
        markdown += f"**Categories**: {len(categorized_data)}\n"
        markdown += f"**PDFPlumber Extraction**: {method_counts.get('pdfplumber', 0)} items\n"
        markdown += f"**Tabular Extraction**: {method_counts.get('tabular', 0)} items\n"
        markdown += f"**Duplicate Policy**: ✅ Fields/Categories allowed, ❌ Values prevented\n\n"
        
        # Collect all items and separate by method
        pdfplumber_items = []
        tabular_items = []
        
        for category, items in categorized_data.items():
            for item_data in items:
                item_data['category'] = category
                if item_data.get('method') == 'pdfplumber':
                    pdfplumber_items.append(item_data)
                elif item_data.get('method') == 'tabular':
                    tabular_items.append(item_data)
        
        # Sort both lists by extraction order
        pdfplumber_items.sort(key=lambda x: x.get('extraction_order', 0))
        tabular_items.sort(key=lambda x: x.get('extraction_order', 0))
        
        headers = ['Category', 'Item', 'Value 1', 'Value 2', 'Value 3', 'Value 4', 'Page']
        json_output_items = []
        
        # PDFPlumber Table
        if pdfplumber_items:
            markdown += f"## 📄 PDFPlumber Extraction Results ({len(pdfplumber_items)} items)\n\n"
            markdown += "| " + " | ".join(headers) + " |\n"
            markdown += "|" + "|".join([":---"] * len(headers)) + "|\n"
            
            for item_data in pdfplumber_items:
                category = item_data['category']
                
                # Clean individual values
                value_1 = str(item_data.get('value_1', '')).strip()
                value_2 = str(item_data.get('value_2', '')).strip()
                value_3 = str(item_data.get('value_3', '')).strip()
                value_4 = str(item_data.get('value_4', '')).strip()
                
                row_data = [
                    category,
                    item_data['item'],
                    value_1,
                    value_2,
                    value_3,
                    value_4,
                    str(item_data.get('page', 'N/A'))
                ]
                
                # Clean data for markdown table - escape pipe characters
                clean_row = [str(cell).replace('|', '\\|').strip() for cell in row_data]
                markdown += "| " + " | ".join(clean_row) + " |\n"
                
                # Add to JSON output
                json_output_items.append({
                    "category": category,
                    "item": item_data['item'],
                    "value_1": value_1,
                    "value_2": value_2,
                    "value_3": value_3,
                    "value_4": value_4,
                    "method": "pdfplumber",
                    "page": item_data.get('page'),
                    "extraction_order": item_data.get('extraction_order', 0)
                })
            
            markdown += "\n"
        
        # Tabular Table
        if tabular_items:
            markdown += f"## 📊 Tabular Extraction Results ({len(tabular_items)} items)\n\n"
            markdown += "| " + " | ".join(headers) + " |\n"
            markdown += "|" + "|".join([":---"] * len(headers)) + "|\n"
            
            for item_data in tabular_items:
                category = item_data['category']
                
                # Clean individual values
                value_1 = str(item_data.get('value_1', '')).strip()
                value_2 = str(item_data.get('value_2', '')).strip()
                value_3 = str(item_data.get('value_3', '')).strip()
                value_4 = str(item_data.get('value_4', '')).strip()
                
                row_data = [
                    category,
                    item_data['item'],
                    value_1,
                    value_2,
                    value_3,
                    value_4,
                    str(item_data.get('page', 'N/A'))
                ]
                
                # Clean data for markdown table - escape pipe characters
                clean_row = [str(cell).replace('|', '\\|').strip() for cell in row_data]
                markdown += "| " + " | ".join(clean_row) + " |\n"
                
                # Add to JSON output
                json_output_items.append({
                    "category": category,
                    "item": item_data['item'],
                    "value_1": value_1,
                    "value_2": value_2,
                    "value_3": value_3,
                    "value_4": value_4,
                    "method": "tabular",
                    "page": item_data.get('page'),
                    "extraction_order": item_data.get('extraction_order', 0)
                })
            
            markdown += "\n"
        
        # Clean JSON output structure
        json_output = {
            "extraction_metadata": {
                "timestamp": datetime.now().isoformat(),
                "total_items": total_items,
                "total_categories": len(categorized_data),
                "model_accuracy": self.classifier.accuracy_score,
                "model_type": self.classifier.model_type,
                "pdfplumber_items": method_counts.get('pdfplumber', 0),
                "tabular_items": method_counts.get('tabular', 0),
                "extraction_methods": ["pdfplumber", "tabular"],
                "duplicate_policy": "Allow duplicate fields/categories, prevent duplicate values"
            },
            "financial_data": json_output_items
        }
        
        return markdown, json_output, accuracy_report


def main():
    """Streamlit application with progress tracking for all operations."""
    st.set_page_config(
        page_title="Financial Data Extractor - Separate Tables", 
        page_icon="🧾", 
        layout="wide"
    )
    
    st.title("🧾 Financial Data Extractor (Separate Tables by Method)")
    st.markdown("*PDFPlumber + Tabular extraction with comprehensive progress tracking*")
    
    with st.sidebar:
        st.header("⚙️ Model Configuration")
        
        model_choice = st.selectbox(
            "Choose Classification Model",
            ('ensemble', 'optimized_rf', 'random_forest', 'gradient_boosting'),
            key='model_selector',
            help="Select the ML model for line classification. Ensemble combines multiple models for best accuracy."
        )
        
        st.header("🔧 Extraction Features")
        st.markdown("""
        - **📄 PDFPlumber**: Text-based extraction (shown first)
        - **📊 Tabular**: Table-based extraction (shown second)
        - **✅ Duplicate Fields**: Same field names allowed
        - **✅ Duplicate Categories**: Same categories allowed
        - **❌ Duplicate Values**: Same value combinations prevented
        - **📈 Progress Tracking**: Real-time status updates
        """)
        
        st.header("📊 Separate Tables Format")
        st.markdown("""
        **Output Structure:**
        - Two distinct tables by extraction method
        - PDFPlumber results shown first
        - Tabular results shown second
        - Method column removed (implicit in headers)
        - Item counts in section headers
        """)
        
        st.header("🧠 Model Features")
        if model_choice == 'ensemble':
            st.markdown("""
            **Ensemble Components:**
            - Random Forest (200 trees)
            - Gradient Boosting (150 estimators)
            - Logistic Regression (L2 regularization)
            - AdaBoost (100 estimators)
            - Extra Trees (150 trees)
            """)
        else:
            st.markdown(f"**{model_choice.replace('_', ' ').title()}** with hyperparameter optimization")
    
    # Initialize or update extractor based on model choice
    if 'extractor' not in st.session_state or st.session_state.get('current_model') != model_choice:
        with st.spinner(f"Initializing {model_choice} model..."):
            st.session_state.extractor = CleanFinancialExtractor(model_type=model_choice)
            st.session_state.current_model = model_choice
        st.success(f"✅ {model_choice} model ready!")
    
    uploaded_file = st.file_uploader(
        "Upload Financial PDF Report", 
        type="pdf",
        help="Upload a PDF containing financial statements, reports, or tabular data"
    )
    
    if uploaded_file:
        st.success(f"✅ File uploaded: **{uploaded_file.name}**")
        
        # Show file details
        file_details = {
            "File Name": uploaded_file.name,
            "File Size": f"{uploaded_file.size / 1024:.2f} KB",
            "File Type": uploaded_file.type
        }
        
        with st.expander("📄 File Details"):
            for key, value in file_details.items():
                st.write(f"**{key}**: {value}")
        
        if st.button("🚀 Run Complete Extraction", type="primary", use_container_width=True):
            start_time = datetime.now()
            
            # Main extraction container
            extraction_container = st.container()
            
            with extraction_container:
                st.info("🎯 Starting comprehensive extraction with progress tracking...")
                
                try:
                    # Extract data using both methods with progress tracking
                    categorized_data = st.session_state.extractor.extract_clean_data(uploaded_file)
                    
                    if categorized_data:
                        # Clear progress indicators
                        st.empty()
                        
                        # Calculate processing time
                        end_time = datetime.now()
                        processing_time = (end_time - start_time).total_seconds()
                        
                        # Generate output with separate tables
                        markdown_output, json_output, accuracy_report = st.session_state.extractor.generate_clean_output(categorized_data)
                        
                        st.success(f"✅ Extraction complete in {processing_time:.2f} seconds!")
                        
                        # Show summary metrics
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Total Items", json_output['extraction_metadata']['total_items'])
                        with col2:
                            st.metric("Categories", json_output['extraction_metadata']['total_categories'])
                        with col3:
                            st.metric("PDFPlumber Items", json_output['extraction_metadata']['pdfplumber_items'])
                        with col4:
                            st.metric("Tabular Items", json_output['extraction_metadata']['tabular_items'])
                        
                        # Display results in tabs
                        tab1, tab2, tab3, tab4 = st.tabs(["📊 Extracted Data", "📈 Model Performance", "💾 Download Options", "🔍 Raw Data"])
                        
                        with tab1:
                            st.markdown(markdown_output)
                        
                        with tab2:
                            st.markdown(accuracy_report)
                            
                            # Additional model insights
                            if hasattr(st.session_state.extractor.classifier, 'classification_report'):
                                with st.expander("📋 Detailed Classification Report"):
                                    st.text(st.session_state.extractor.classifier.classification_report)
                        
                        with tab3:
                            col1, col2 = st.columns(2)
                            with col1:
                                st.download_button(
                                    label="📥 Download Markdown Report",
                                    data=markdown_output,
                                    file_name=f"financial_report_separate_tables_{model_choice}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                                    mime="text/markdown"
                                )
                            with col2:
                                st.download_button(
                                    label="📥 Download JSON Data",
                                    data=json.dumps(json_output, indent=2),
                                    file_name=f"financial_data_separate_tables_{model_choice}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                                    mime="application/json"
                                )
                            
                            # Additional download options
                            st.subheader("📈 Additional Exports")
                            
                            # Convert to DataFrame for CSV export
                            df = pd.DataFrame(json_output['financial_data'])
                            csv_data = df.to_csv(index=False)
                            
                            st.download_button(
                                label="📊 Download CSV Data",
                                data=csv_data,
                                file_name=f"financial_data_separate_tables_{model_choice}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv"
                            )
                        
                        with tab4:
                            st.subheader("🔍 Raw JSON Structure")
                            st.json(json_output)
                            
                            st.subheader("📊 Data Statistics")
                            st.write("**Extraction Summary:**")
                            st.write(f"- Processing Time: {processing_time:.2f} seconds")
                            st.write(f"- Model Accuracy: {json_output['extraction_metadata']['model_accuracy']:.3f}")
                            st.write(f"- Extraction Methods: {', '.join(json_output['extraction_metadata']['extraction_methods'])}")
                            
                    else:
                        st.warning("⚠️ No financial data could be extracted from this document. Please ensure the PDF contains tabular financial information.")
                        
                        # Troubleshooting suggestions
                        with st.expander("🔧 Troubleshooting Suggestions"):
                            st.markdown("""
                            **Common Issues and Solutions:**
                            1. **Empty Results**: PDF may contain only images or non-selectable text
                            2. **Poor Quality**: Try a different model type from the sidebar
                            3. **Complex Layout**: Document structure may be too complex for automatic parsing
                            4. **Protected PDF**: Ensure PDF is not password-protected or encrypted
                            5. **Language Issues**: Model works best with English financial documents
                            
                            **Try These Steps:**
                            - Use a different model type (try 'ensemble' for best results)
                            - Ensure PDF text is selectable (not scanned images)
                            - Check if PDF contains structured tabular data
                            - Verify the document is a financial report with numerical data
                            """)
                        
                except Exception as e:
                    st.error(f"❌ Extraction failed: {str(e)}")
                    with st.expander("🔍 Error Details"):
                        st.exception(e)
                        st.markdown("**Troubleshooting Tips:**")
                        st.markdown("""
                        - Ensure the PDF is not password-protected
                        - Check that the PDF contains readable text (not just images)
                        - Try a different model type from the sidebar
                        - Verify the PDF has structured financial data
                        - Check file size (very large files may cause memory issues)
                        """)
    else:
        st.info("👆 Please upload a PDF file to begin extraction")
        
        # Show example with separate tables
        with st.expander("📋 Separate Tables Format Preview"):
            st.markdown("""
            ### Sample Extraction Result (Separate Tables by Method)
            
            ## 📄 PDFPlumber Extraction Results (3 items)
            | Category | Item | Value 1 | Value 2 | Value 3 | Value 4 | Page |
            |:---------|:-----|:--------|:--------|:--------|:--------|:-----|
            | Revenue | Product Sales | $125,000.00 | $120,000.00 | | | 1 |
            | Revenue | Service Revenue | $75,500.00 | | | | 1 |
            | Expenses | Marketing Costs | ($15,000.00) | ($12,000.00) | | | 2 |
            
            ## 📊 Tabular Extraction Results (4 items)
            | Category | Item | Value 1 | Value 2 | Value 3 | Value 4 | Page |
            |:---------|:-----|:--------|:--------|:--------|:--------|:-----|
            | Revenue | Product Sales | $125,000.00 | $120,000.00 | $130,000.00 | | 1 |
            | Revenue | Product Sales | $135,000.00 | $125,000.00 | | | 1 |
            | Expenses | Marketing Costs | ($18,000.00) | ($16,000.00) | | | 2 |
            | Revenue | Other Income | $5,250.75 | $4,800.00 | | | 2 |
            
            **Key Features:**
            - 📄 PDFPlumber results in first table
            - 📊 Tabular results in second table  
            - ✅ Same field names allowed in both tables
            - ✅ Same categories allowed in both tables
            - ❌ Identical value combinations prevented within each method
            - 📊 Separate item counts for each method
            - 📈 Real-time progress tracking during extraction
            """)


if __name__ == "__main__":
    main()