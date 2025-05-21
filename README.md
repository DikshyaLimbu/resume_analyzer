# Resume Matcher Pro

A modern resume matching system that helps recruiters and hiring managers find the best candidates for their job openings. The application uses NLP techniques to analyze resumes and match them against job descriptions.

## ✨ What Has Been Implemented

### 1. Resume Text Extraction
- A reusable function built to extract text from uploaded PDFs using multiple engines:
  - PyMuPDF (fitz)
  - pdfplumber
  - pdfminer.six
- Ensures fallback reliability when parsing various PDF formats

### 2. Resume Section Parsing
- Regex-based parser extracts:
  - Email
  - Phone
- skillNer extracts:
  - skills
- SBERT extracts key sections:
  - Education
  - Experience
  - Profile
- These sections are used for both display and context-aware matching

### 3. Data Cleaning and Preprocessing
- Cleaned resume text using:
  - ASCII filtering
  - Punctuation and newline removal
  - Normalization (lowercasing, spacing)
- Applied to both real resume PDFs and the labeled dataset

### 4. Conversational Interface
- Flask Web App implementation with:
  - PDF upload functionality
  - Job description input
  - Display of top matching resumes with similarity scores
  - Key information display

### 5. Job Description Matching
- Implementation includes:
  - Text-based similarity comparison using SBERT
  - Section-specific analysis
  - Ranked list of best-fit resumes

## 📦 Project Structure

```
Model_and_Interface/
├── app.py                      # Flask endpoint
├── resume_model.py            # Resume processing logic
├── requirements.txt           # Project dependencies
├── uploads/                   # Storage for web app file uploads
├── model/                     # Model storage
├── templates/                 # HTML templates
│   ├── base.html             # Base template
│   ├── index.html            # Upload form
│   ├── results.html          # Matching results
│   ├── analytics.html        # Analytics dashboard
│   ├── 404.html             # Not found error page
│   └── 500.html             # Server error page
└── README.md                 # Documentation
```

## 🚀 Getting Started

### Prerequisites
- Python 3.13 or higher
- pip package manager
- Visual Studio Build Tools (for Windows users)

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
```

2. Install core dependencies:
```bash
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

3. Install NLP components:
```bash
# Install NLTK and download required data
pip install nltk
python -m nltk.downloader punkt averaged_perceptron_tagger

# Install TextBlob for additional NLP capabilities
pip install textblob
```

4. Install PDF processing dependencies:
```bash
pip install PyMuPDF pdfplumber pdfminer.six PyPDF2
```

5. Install web framework components:
```bash
pip install Flask Werkzeug plotly
```

6. Install data processing libraries:
```bash
pip install numpy pandas scikit-learn openpyxl xlrd
```

7. Install model packages:
```bash
pip install sentence-transformer
```

### Directory Setup

Create necessary directories:
```bash
mkdir -p uploads model templates
```

### Running the Application

Start the Flask application:
```bash
python app.py
```

The application will be available at http://localhost:5000

## 🔧 Usage

1. **Upload Resumes**: 
   - Select one or more PDF resumes
   - System automatically processes and extracts information

2. **Job Description Matching**:
   - Enter the job description text
   - System ranks resumes by relevance
   - View detailed resume sections and match scores

3. **Analytics**:
   - Access the analytics dashboard
   - View resume statistics and distributions
   - Export data for further analysis

## 🛠️ Technical Implementation Details

### PDF Processing
- Multiple extraction methods for reliability
- Fallback system for different PDF formats
- Robust text extraction pipeline

### Text Analysis
- Section-based parsing with keyword recognition
- Regex-based section extraction
- Context-aware content analysis

### Web Framework
- Flask with Jinja2 templating
- Bootstrap 5 with responsive design
- FontAwesome icons integration

### Data Management
- JSON and Excel export options
- Structured data storage
- Analytics data processing

## 📊 Outputs and Files

- `parsed_resumes.xlsx`: Structured resume data
- `parsed_resumes.json`: JSON format resume data
- Analytics visualizations
- Export functionality for analysis results

## 🔄 Current Development

- Enhanced analytics visualizations
- Batch processing improvements
- API endpoint documentation
- Performance optimizations

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.
