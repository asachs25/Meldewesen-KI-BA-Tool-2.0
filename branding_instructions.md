# 🎨 Corporate Branding Instructions

## 📋 Files to Replace/Update

### 1. **Frontend Files** (Replace Completely)
```
landing.html    # Professional landing page
index.html      # Main analyzer interface  
main.py         # Enhanced backend with corporate features
```

### 2. **Environment Configuration** (Create New)
```
.env           # Corporate configuration file
```

### 3. **Branding Assets** (Add Your Own)
```
static/logo.svg           # Your organization's logo
static/favicon.ico        # Your organization's favicon
static/brand-assets/      # Additional brand assets
```

---

## 🏢 Step-by-Step Branding Process

### Step 1: Update Environment Variables
Create a `.env` file in your project root:

```bash
# Corporate Branding Configuration
CORPORATE_APP_NAME="Your Business Intelligence Platform"
CORPORATE_ORGANIZATION="Your Organization Name"
CORPORATE_LOGO_URL="/static/your-logo.svg"
CORPORATE_PRIMARY_COLOR="#YOUR_HEX_COLOR"
CORPORATE_SECONDARY_COLOR="#YOUR_HEX_COLOR"
CORPORATE_SUPPORT_EMAIL="support@yourcompany.com"
CORPORATE_DOCS_URL="/docs"

# Technical Configuration
OLLAMA_HOST=http://localhost:11434
DEFAULT_MODEL=llama3
APP_HOST=0.0.0.0
APP_PORT=8000
DEBUG=True
MAX_CONTEXT_DOCS=6
MAX_FILE_SIZE=10485760
```

### Step 2: Replace Logo References
In both `landing.html` and `index.html`, find and replace:

**Find:**
```html
<img src="https://upload.wikimedia.org/wikipedia/commons/5/56/Deloitte.svg" alt="Professional Consulting" class="brand-logo">
```

**Replace with:**
```html
<img src="{{YOUR_LOGO_URL}}" alt="{{YOUR_ORGANIZATION}}" class="brand-logo">
```

### Step 3: Update Color Scheme
In the CSS sections of both HTML files, update the CSS variables:

**Find:**
```css
:root {
    --brand-primary: #86bc25;      /* Primary brand color */
    --brand-secondary: #0d2818;    /* Secondary brand color */
    --brand-accent: #d4edda;       /* Accent color */
}
```

**Replace with your colors:**
```css
:root {
    --brand-primary: #YOUR_PRIMARY_COLOR;
    --brand-secondary: #YOUR_SECONDARY_COLOR;
    --brand-accent: #YOUR_ACCENT_COLOR;
}
```

### Step 4: Update Text Content
Replace placeholder text in both HTML files:

**Landing Page (`landing.html`):**
- Replace "Professional Consulting" with your organization name
- Update the hero title and descriptions
- Modify the footer content
- Update contact information

**Analyzer Interface (`index.html`):**
- Replace "Professional Consulting" with your organization name  
- Update the brand title and subtitle
- Modify navigation labels if needed

### Step 5: Backend Configuration
The `main.py` file automatically reads from your `.env` file, so no manual changes needed if you've configured the environment variables correctly.

---

## 🎨 Brand Customization Options

### Color Schemes
Choose colors that match your corporate identity:

```css
/* Conservative/Professional */
--brand-primary: #003366;     /* Navy blue */
--brand-secondary: #1a1a1a;   /* Dark gray */
--brand-accent: #e6f3ff;      /* Light blue */

/* Modern/Tech */
--brand-primary: #6366f1;     /* Indigo */
--brand-secondary: #1e293b;   /* Slate */
--brand-accent: #e0e7ff;      /* Light indigo */

/* Financial/Corporate */
--brand-primary: #059669;     /* Green */
--brand-secondary: #064e3b;   /* Dark green */
--brand-accent: #d1fae5;      /* Light green */
```

### Typography Options
Update font families in the CSS:

```css
/* Professional */
font-family: 'Source Sans Pro', -apple-system, BlinkMacSystemFont, sans-serif;

/* Modern */
font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;

/* Corporate */
font-family: 'Roboto', -apple-system, BlinkMacSystemFont, sans-serif;
```

---

## 📁 File Structure After Branding

```
your-project/
├── 🌐 Frontend (Updated)
│   ├── landing.html           # ✅ Corporate landing page
│   ├── index.html             # ✅ Corporate analyzer interface
│   └── static/
│       ├── your-logo.svg      # 🆕 Your organization's logo
│       ├── favicon.ico        # 🆕 Your favicon
│       └── brand-assets/      # 🆕 Additional brand files
├── 🔧 Backend (Updated)
│   ├── main.py               # ✅ Corporate-enhanced backend
│   └── pages/
│       ├── rag_cag.py        # ✅ Document processing
│       └── data_analytics.py  # ✅ SQL generation
├── ⚙️ Configuration (New)
│   ├── .env                  # 🆕 Corporate configuration
│   └── requirements.txt      # ✅ Dependencies
└── 📚 Documents
    ├── rag_docs/             # ✅ Guidelines & regulations
    ├── cag_docs/             # ✅ Company-specific rules
    └── mapping_docs/         # ✅ Technical mapping files
```

---

## 🧪 Testing Your Branding

### 1. Visual Verification
- [ ] Logo displays correctly in header
- [ ] Colors match your brand guidelines
- [ ] Text content reflects your organization
- [ ] Favicon appears in browser tab

### 2. Functional Testing
- [ ] All pages load without errors
- [ ] Navigation works correctly
- [ ] Corporate config API returns correct values
- [ ] Upload and analysis functions work

### 3. Brand Consistency Check
- [ ] Landing page matches analyzer interface
- [ ] Color scheme is consistent throughout
- [ ] Typography is uniform
- [ ] Messaging aligns with brand voice

---

## 🔧 Advanced Customization

### Custom CSS Themes
Create additional CSS files for different themes:

```css
/* themes/dark-corporate.css */
:root {
    --brand-primary: #your-color;
    --bg-primary: #1a1a1a;
    --text-primary: #ffffff;
}

/* themes/light-corporate.css */
:root {
    --brand-primary: #your-color;
    --bg-primary: #ffffff;
    --text-primary: #1a1a1a;
}
```

### Custom Components
Add organization-specific components:

```html
<!-- Corporate disclaimer -->
<div class="corporate-disclaimer">
    <p>© 2024 Your Organization. Enterprise Business Intelligence Platform.</p>
    <p>For internal use only. Confidential and proprietary.</p>
</div>

<!-- Department selector -->
<select id="department-selector">
    <option value="risk">Risk Management</option>
    <option value="compliance">Compliance</option>
    <option value="analytics">Business Analytics</option>
</select>
```

### Integration with Corporate Systems
Extend the backend for enterprise integration:

```python
# Add to main.py for SSO integration
@app.middleware("http")
async def corporate_auth_middleware(request: Request, call_next):
    # Add your SSO/authentication logic here
    return await call_next(request)

# Add for department-specific analysis
@app.post("/api/analyze/department/{dept_id}")
async def department_specific_analysis(dept_id: str, request: AnalysisRequest):
    # Department-specific logic
    pass
```

---

## 📞 Support & Troubleshooting

### Common Issues

**Issue**: Logo not displaying
- **Solution**: Check file path and ensure logo file exists in `/static/` directory

**Issue**: Colors not updating  
- **Solution**: Clear browser cache and verify CSS variable names are correct

**Issue**: Environment variables not loading
- **Solution**: Ensure `.env` file is in project root and restart the application

**Issue**: Corporate config not working
- **Solution**: Verify all environment variables are set and restart backend

### Getting Help

1. **Check logs**: Look at `analyzer.log` for error messages
2. **Verify configuration**: Use `/api/corporate/config` endpoint to check settings
3. **Test API**: Visit `/docs` for interactive API documentation
4. **Health check**: Use `/api/health` to verify system status

---

## ✅ Branding Checklist

### Pre-Launch
- [ ] All environment variables configured
- [ ] Logo and favicon added to static directory
- [ ] Colors updated to match brand guidelines
- [ ] Text content customized for organization
- [ ] Contact information updated
- [ ] Testing completed successfully

### Post-Launch
- [ ] User feedback collected
- [ ] Performance monitoring enabled
- [ ] Security review completed
- [ ] Documentation updated
- [ ] Training materials prepared

---

## 🚀 Next Steps

1. **Deploy**: Follow deployment guide for your environment
2. **Monitor**: Set up logging and monitoring
3. **Scale**: Configure for multiple users/departments
4. **Enhance**: Add organization-specific features
5. **Maintain**: Regular updates and security patches

Your professional, branded Business Intelligence Platform is now ready for enterprise deployment!