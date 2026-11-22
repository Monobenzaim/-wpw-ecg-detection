# 🔑 Groq API Setup Guide

Complete guide to setting up Groq's LLM API for AI-powered medical explanations in the WPW ECG Detection System.

## 📋 Table of Contents

- [What is Groq?](#what-is-groq)
- [Why Use Groq?](#why-use-groq)
- [Getting an API Key](#getting-an-api-key)
- [Configuration Methods](#configuration-methods)
- [Testing Your Setup](#testing-your-setup)
- [Usage Examples](#usage-examples)
- [Troubleshooting](#troubleshooting)
- [Best Practices](#best-practices)

---

## What is Groq?

**Groq** is an AI infrastructure company providing ultra-fast LLM inference. Their API gives you access to powerful language models including:

- **Llama 3.1 70B** (default in this project)
- **Llama 3.1 8B** (faster, slightly less capable)
- **Mixtral 8x7B**
- **Gemma 7B**

### Key Features:
- ⚡ **Extremely fast** inference (10-100x faster than typical cloud LLMs)
- 🆓 **Free tier** with generous limits
- 🔒 **Secure** API with standard authentication
- 📊 **High quality** outputs for medical text generation

---

## Why Use Groq?

### For This Project:

1. **Free & Fast**: No cost for development and research
2. **Medical Explanations**: Llama 3.1 70B provides high-quality medical text
3. **Low Latency**: Fast response times for interactive web app
4. **Easy Integration**: Simple REST API with Python SDK
5. **No Credit Card**: Sign up without payment info

### Comparison:

| Provider | Free Tier | Speed | Quality | Medical Use |
|----------|-----------|-------|---------|-------------|
| Groq | ✅ Generous | ⚡⚡⚡ Ultra Fast | ⭐⭐⭐⭐ | ✅ Excellent |
| OpenAI | ❌ Paid only | ⚡⚡ Fast | ⭐⭐⭐⭐⭐ | ✅ Excellent |
| Anthropic | ❌ Paid only | ⚡⚡ Fast | ⭐⭐⭐⭐⭐ | ✅ Excellent |
| Local LLMs | ✅ Free | ⚡ Slow | ⭐⭐⭐ Variable | ⚠️ Depends |

---

## Getting an API Key

### Step-by-Step Guide:

#### 1. Visit Groq Console

Go to: **[https://console.groq.com/keys](https://console.groq.com/keys)**

#### 2. Sign Up / Log In

- Click **"Sign Up"** or **"Log In"**
- Options:
  - Google account
  - GitHub account
  - Email/password

**No credit card required!** ✅

#### 3. Create API Key

Once logged in:

1. Navigate to **"API Keys"** section (usually auto-redirected)
2. Click **"Create API Key"**
3. Give it a name (e.g., "WPW ECG Project")
4. Click **"Create"**

#### 4. Copy Your Key

⚠️ **IMPORTANT**: Copy your API key immediately!

```
gsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

You **won't be able to see it again** after closing the dialog.

If you lose it:
- Revoke the old key
- Create a new one

#### 5. Secure Storage

**DO NOT:**
- ❌ Commit to Git
- ❌ Share publicly
- ❌ Post in screenshots
- ❌ Email without encryption

**DO:**
- ✅ Store in `.env` file (gitignored)
- ✅ Use environment variables
- ✅ Rotate regularly
- ✅ Revoke if compromised

---

## Configuration Methods

### Method 1: Environment Variable (Recommended)

**Best for**: Local development, persistent configuration

```bash
# Copy template
cp .env.example .env

# Edit .env file
nano .env  # or VS Code, notepad, etc.

# Add your key
GROQ_API_KEY=gsk_your_actual_key_here

# Save and exit
```

**Verify:**
```bash
# On Linux/Mac
echo $GROQ_API_KEY

# On Windows PowerShell
echo $env:GROQ_API_KEY
```

**Usage in Python:**
```python
import os
from groq import Groq

api_key = os.getenv("GROQ_API_KEY")
client = Groq(api_key=api_key)
```

### Method 2: Streamlit Secrets

**Best for**: Streamlit Cloud deployment

```bash
# Create secrets file
mkdir -p .streamlit
nano .streamlit/secrets.toml

# Add:
GROQ_API_KEY = "gsk_your_actual_key_here"
```

**Usage in Streamlit:**
```python
import streamlit as st

api_key = st.secrets["GROQ_API_KEY"]
```

### Method 3: Web Interface (Session Only)

**Best for**: Quick testing, demos

1. Launch app: `streamlit run app_enhanced.py`
2. In sidebar, find **"🔑 Groq API Settings"**
3. Paste API key into text input
4. API key valid for current session only

**Note**: Not saved permanently, re-enter after restart.

### Method 4: System Environment Variable

**Best for**: Global configuration across projects

**On Linux/Mac:**
```bash
# Add to ~/.bashrc or ~/.zshrc
export GROQ_API_KEY="gsk_your_actual_key_here"

# Reload
source ~/.bashrc
```

**On Windows:**
```powershell
# PowerShell (Admin)
[System.Environment]::SetEnvironmentVariable(
    "GROQ_API_KEY",
    "gsk_your_actual_key_here",
    "User"
)

# Or use System Properties > Environment Variables GUI
```

---

## Testing Your Setup

### Quick Test Script

```python
# test_groq.py
import os
from groq import Groq

# Get API key
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    print("❌ GROQ_API_KEY not found!")
    print("Set it in .env file or environment")
    exit(1)

print(f"✅ API Key found: {api_key[:20]}...")

# Test connection
try:
    client = Groq(api_key=api_key)

    response = client.chat.completions.create(
        model="llama-3.1-70b-versatile",
        messages=[
            {"role": "user", "content": "Say 'Hello from Groq!' if you're working."}
        ],
        max_tokens=50
    )

    print(f"✅ Groq API working!")
    print(f"Response: {response.choices[0].message.content}")

except Exception as e:
    print(f"❌ Error: {e}")
```

**Run:**
```bash
python test_groq.py
```

**Expected Output:**
```
✅ API Key found: gsk_abcdefghij123456...
✅ Groq API working!
Response: Hello from Groq!
```

### Test in Web App

1. Launch: `streamlit run app_enhanced.py`
2. Configure API key in sidebar
3. Make a prediction with any ECG data
4. Look for **"🤖 AI-Powered Medical Analysis"** section
5. If explanation appears, it's working! ✅

---

## Usage Examples

### Basic Medical Explanation

```python
from src.llm_explainer import ECGExplainer

# Initialize
explainer = ECGExplainer(api_key="your_key_here")

# Generate explanation
explanation = explainer.explain_prediction(
    features={'age': 45, 'hr_mean': 128, ...},
    prediction=1,  # WPW detected
    probability=0.87,
    risk_level='high',
    model_name='Random Forest'
)

print(explanation)
```

### Custom Prompts

```python
from groq import Groq

client = Groq(api_key="your_key_here")

response = client.chat.completions.create(
    model="llama-3.1-70b-versatile",
    messages=[
        {
            "role": "system",
            "content": "You are a cardiologist."
        },
        {
            "role": "user",
            "content": "Explain WPW syndrome in simple terms."
        }
    ],
    temperature=0.7,
    max_tokens=500
)

print(response.choices[0].message.content)
```

### Streaming Responses

```python
response = client.chat.completions.create(
    model="llama-3.1-70b-versatile",
    messages=[{"role": "user", "content": "Explain..."}],
    stream=True
)

for chunk in response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

---

## Troubleshooting

### Error: "API key not found"

**Solution:**
```bash
# Check .env exists
ls -la .env

# Verify content
cat .env | grep GROQ

# Restart terminal/app after setting
```

### Error: "Invalid API key"

**Causes:**
- Incorrect key (copy/paste error)
- Spaces before/after key
- Key revoked

**Solution:**
- Generate new key at console.groq.com
- Copy carefully (no extra characters)
- Update `.env` file
- Restart application

### Error: "Rate limit exceeded"

**Free Tier Limits:**
- ~30 requests per minute
- ~14,400 requests per day

**Solutions:**
- Wait a few minutes
- Reduce request frequency
- Upgrade to paid tier (if needed)
- Implement caching

### Error: "Model not found"

**Solution:**
Check model name:
- ✅ `llama-3.1-70b-versatile`
- ✅ `llama-3.1-8b-instant`
- ❌ `llama-3.1-70b` (wrong)

### Slow Responses

**Groq is known for speed**, so if slow:
- Check internet connection
- Try different model (8B is faster)
- Reduce `max_tokens`
- Check for API issues: [status.groq.com](https://status.groq.com)

### No AI Explanations in App

**Checklist:**
1. ✅ API key configured?
2. ✅ Sidebar shows "✅ API key configured"?
3. ✅ Prediction completed successfully?
4. ✅ Check browser console for errors (F12)
5. ✅ Internet connection active?

---

## Best Practices

### Security

```python
# ✅ GOOD - Use environment variable
api_key = os.getenv("GROQ_API_KEY")

# ❌ BAD - Hardcoded key
api_key = "gsk_12345..."  # NEVER DO THIS!
```

### Error Handling

```python
from src.llm_explainer import ECGExplainer

try:
    explainer = ECGExplainer(api_key=api_key)
    explanation = explainer.explain_prediction(...)
except Exception as e:
    print(f"LLM unavailable: {e}")
    # Fallback to basic explanation
    explanation = "LLM service temporarily unavailable."
```

### Rate Limiting

```python
import time

# Add delay between requests
time.sleep(2)  # 2 second delay

# Or implement proper rate limiting
from ratelimit import limits, sleep_and_retry

@sleep_and_retry
@limits(calls=20, period=60)  # 20 calls per minute
def call_groq_api():
    # Your API call here
    pass
```

### Cost Optimization

**Free tier is generous**, but to maximize:

1. **Cache responses** for identical inputs
2. **Reduce max_tokens** to minimum needed
3. **Use 8B model** for simpler tasks
4. **Batch similar requests**
5. **Implement fallbacks** for non-critical features

### Production Deployment

For production apps:

```python
# Use secrets management
import boto3  # AWS Secrets Manager

def get_groq_key():
    client = boto3.client('secretsmanager')
    response = client.get_secret_value(SecretId='groq-api-key')
    return response['SecretString']

api_key = get_groq_key()
```

---

## API Limits & Pricing

### Free Tier (as of 2026)

- **Requests per minute**: ~30
- **Requests per day**: ~14,400
- **Context window**: 8K-128K tokens (model dependent)
- **Models**: All models available
- **Support**: Community

**Perfect for:**
- Development
- Research
- Academic projects
- Small-scale applications

### Paid Tiers

Check latest pricing: [groq.com/pricing](https://groq.com/pricing)

---

## Additional Resources

- **Groq Console**: [console.groq.com](https://console.groq.com)
- **Documentation**: [console.groq.com/docs](https://console.groq.com/docs)
- **API Reference**: [console.groq.com/docs/api-reference](https://console.groq.com/docs/api-reference)
- **Status Page**: [status.groq.com](https://status.groq.com)
- **Community**: [discord.gg/groq](https://discord.gg/groq)

---

## Summary

✅ **Quick Setup Checklist:**

1. [ ] Get API key from [console.groq.com/keys](https://console.groq.com/keys)
2. [ ] Copy key to `.env` file: `GROQ_API_KEY=your_key`
3. [ ] Test with `python test_groq.py`
4. [ ] Launch app: `streamlit run app_enhanced.py`
5. [ ] Verify AI explanations appear
6. [ ] Keep key secure (never commit to git)

**You're ready to generate AI-powered medical explanations!** 🎉

---

**Need help?** Open an issue on GitHub or check the main [README.md](README.md).
