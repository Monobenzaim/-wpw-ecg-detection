# 🔒 Security Guidelines

Security best practices for the WPW ECG Detection System. This project handles medical data and API keys - follow these guidelines carefully.

---

## ⚠️ CRITICAL: What NEVER to Commit

### 🔑 API Keys & Secrets

**NEVER commit these to Git:**

- Groq API keys (`gsk_...`)
- Any API keys or tokens
- `.env` files with secrets
- `secrets.toml` files
- Database credentials
- SSH private keys
- AWS/Azure/GCP credentials

### 🏥 Patient Data

**NEVER commit:**

- Real patient names
- Patient IDs (use anonymized/fake for demos)
- Medical record numbers (MRN)
- Personal health information (PHI)
- Generated PDF reports with real data
- Any HIPAA-protected information

### 📊 Large/Generated Files

**NEVER commit:**

- Trained model files (`*.pkl` - too large)
- Large datasets (`*.csv` with thousands of rows)
- Generated reports (`reports/*.pdf`)
- Benchmark results (`*.png`, `*.csv` - regeneratable)

---

## ✅ What's Safe to Commit

- Source code (`*.py`)
- Documentation (`*.md`)
- Configuration templates (`.env.example`)
- Requirements (`requirements.txt`)
- `.gitignore` itself
- Notebook files (`*.ipynb`) - if outputs cleared
- README files
- Example/sample data (anonymized, small)

---

## 🛡️ Security Layers

### Layer 1: `.gitignore`

**Purpose**: Prevent accidental commits

The `.gitignore` file is configured to block:

```gitignore
# API Keys
.env
.env.local
**/GROQ_API_KEY*
**/groq_api_key*
credentials.json

# Patient Data
reports/*.pdf
reports/*.json

# Models
models/*.pkl
models/*.joblib

# Large Data
data/*.csv
data/*.zip
```

**Verify before commit:**
```bash
git status
git diff

# Check what would be committed
git add --dry-run .
```

### Layer 2: Environment Variables

**Use `.env` files** (never commit!):

```bash
# .env
GROQ_API_KEY=gsk_your_actual_key_here
```

**Load in Python:**
```python
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
```

**Advantages:**
- ✅ Not in source code
- ✅ Easy to rotate keys
- ✅ Different keys per environment
- ✅ Automatically gitignored

### Layer 3: Secrets Management (Production)

For production deployment, use proper secrets management:

**Cloud Options:**
- **AWS**: Secrets Manager, Parameter Store
- **Azure**: Key Vault
- **GCP**: Secret Manager
- **Streamlit Cloud**: Streamlit Secrets

**Example (AWS Secrets Manager):**
```python
import boto3
import json

def get_secret(secret_name):
    client = boto3.client('secretsmanager')
    response = client.get_secret_value(SecretId=secret_name)
    return json.loads(response['SecretString'])

secrets = get_secret('wpw-ecg-prod-keys')
groq_key = secrets['GROQ_API_KEY']
```

---

## 🔐 API Key Management

### Getting Keys

1. **Groq**: [console.groq.com/keys](https://console.groq.com/keys)
2. Create key with descriptive name
3. Copy immediately (shown only once)
4. Store securely

### Storing Keys

**✅ GOOD:**
```python
# .env file
GROQ_API_KEY=gsk_abc123...

# Python
import os
api_key = os.getenv("GROQ_API_KEY")
```

**❌ BAD:**
```python
# NEVER hardcode!
api_key = "gsk_abc123..."  # DON'T DO THIS!
```

### Rotating Keys

**When to rotate:**
- Regularly (every 90 days)
- If key is compromised
- When team member leaves
- After security incident

**How to rotate:**
1. Generate new key in Groq console
2. Update `.env` file
3. Test application
4. Revoke old key
5. Update documentation

### Key Compromise Response

**If you accidentally commit a key:**

1. **IMMEDIATELY revoke it**
   - Go to Groq console
   - Delete the compromised key
   - Generate new key

2. **Remove from Git history**
   ```bash
   # Use BFG Repo-Cleaner or git-filter-repo
   # This is complex - prevention is better!

   # Simple but destructive approach:
   git reset --hard HEAD~1  # if just committed
   git push --force origin main  # WARNING: Destructive!
   ```

3. **Update everywhere**
   - New key in `.env`
   - Update in production
   - Notify team

4. **Review and learn**
   - How did it happen?
   - Update `.gitignore`
   - Add pre-commit hooks

---

## 🏥 Patient Data Protection

### HIPAA Compliance Considerations

If handling real patient data, ensure:

#### Technical Safeguards

1. **Encryption**
   - Data at rest: AES-256
   - Data in transit: TLS 1.2+
   - Database encryption
   - Encrypted backups

2. **Access Controls**
   - Authentication required
   - Role-based access (RBAC)
   - Audit logging
   - Session timeouts

3. **Data Minimization**
   - Collect only necessary data
   - Anonymize when possible
   - Set retention policies
   - Secure deletion

#### Administrative Safeguards

1. **Policies**
   - Data handling procedures
   - Incident response plan
   - Regular risk assessments
   - Staff training

2. **Documentation**
   - Access logs
   - Security measures
   - Policy compliance
   - Incident reports

### For Development/Testing

**Use synthetic data:**

```python
# Example: Fake patient data
patient_info = {
    'name': 'John Doe',         # Fake name
    'id': 'DEMO-12345',         # Demo ID
    'age': 45,
    'sex': 'Male'
}
```

**Label clearly:**
```python
# In reports
header = "DEMONSTRATION REPORT - NOT FOR CLINICAL USE"
```

**Anonymize real data:**
```python
# Remove identifying fields
df = df.drop(['patient_name', 'mrn', 'ssn'], axis=1)

# Hash IDs
import hashlib
df['patient_id'] = df['patient_id'].apply(
    lambda x: hashlib.sha256(str(x).encode()).hexdigest()[:10]
)
```

---

## 📊 Data Security

### Dataset Protection

```gitignore
# .gitignore

# Exclude all data
data/*.csv
data/*.xlsx
data/*.zip

# Except examples
!data/sample_*.csv
!data/example_*.csv
```

### Database Security (If Using)

```python
# Use parameterized queries (prevent SQL injection)
cursor.execute(
    "SELECT * FROM patients WHERE id = %s",
    (patient_id,)  # Safe: parameterized
)

# NEVER:
cursor.execute(
    f"SELECT * FROM patients WHERE id = {patient_id}"  # UNSAFE!
)
```

### File Permissions

```bash
# Restrict access to sensitive files
chmod 600 .env              # Owner read/write only
chmod 600 credentials.json
chmod 700 models/           # Owner full access only
```

---

## 🔍 Security Checklist

### Before Each Commit

- [ ] Run `git status` - review all changes
- [ ] Run `git diff` - check actual content
- [ ] No API keys in code
- [ ] No real patient data
- [ ] No `.env` file staged
- [ ] No large model files
- [ ] Documentation updated

### Before Deployment

- [ ] All secrets in environment variables
- [ ] `.env.example` updated (without actual keys)
- [ ] Security headers configured
- [ ] HTTPS enabled
- [ ] Authentication implemented
- [ ] Logging configured
- [ ] Backup strategy in place
- [ ] Incident response plan ready

### Regular Maintenance

- [ ] Rotate API keys (quarterly)
- [ ] Update dependencies (`pip list --outdated`)
- [ ] Review access logs
- [ ] Audit user permissions
- [ ] Test backup restoration
- [ ] Review security advisories

---

## 🚨 Incident Response

### If Security Breach Occurs

1. **Immediate Actions**
   - Revoke compromised credentials
   - Block unauthorized access
   - Preserve evidence/logs
   - Document timeline

2. **Assessment**
   - Determine scope
   - Identify affected data
   - Assess impact
   - Find root cause

3. **Notification**
   - Notify stakeholders
   - Inform affected parties (if PHI involved)
   - Report to authorities (if required)
   - Update team

4. **Remediation**
   - Fix vulnerability
   - Update systems
   - Enhance monitoring
   - Review and improve

---

## 🛠️ Security Tools

### Pre-commit Hooks

Prevent committing secrets:

```bash
# Install
pip install pre-commit

# Configure (.pre-commit-config.yaml)
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.4.0
    hooks:
      - id: detect-private-key
      - id: check-added-large-files
      - id: check-yaml

  - repo: https://github.com/Yelp/detect-secrets
    rev: v1.4.0
    hooks:
      - id: detect-secrets

# Install hooks
pre-commit install

# Now hooks run automatically before each commit
```

### Secrets Scanning

**GitHub:**
- Enable secret scanning in repo settings
- Automatic detection of leaked keys
- Notifications if secrets found

**git-secrets (AWS)**:
```bash
git clone https://github.com/awslabs/git-secrets
cd git-secrets
make install

# Setup
git secrets --install
git secrets --register-aws
```

### Dependency Scanning

```bash
# Check for vulnerabilities
pip install safety
safety check

# Or
pip install pip-audit
pip-audit
```

---

## 📚 Compliance Resources

### HIPAA (US Healthcare)

- [HHS HIPAA Homepage](https://www.hhs.gov/hipaa/)
- [HIPAA Security Rule](https://www.hhs.gov/hipaa/for-professionals/security/)
- [HIPAA Privacy Rule](https://www.hhs.gov/hipaa/for-professionals/privacy/)

### GDPR (EU Data Protection)

- [GDPR Official Text](https://gdpr-info.eu/)
- [Data Protection Principles](https://gdpr-info.eu/art-5-gdpr/)
- [Right to Erasure](https://gdpr-info.eu/art-17-gdpr/)

### General Best Practices

- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [NIST Cybersecurity Framework](https://www.nist.gov/cyberframework)
- [CIS Controls](https://www.cisecurity.org/controls)

---

## ✅ Current Project Protections

This project already includes:

- ✅ Comprehensive `.gitignore`
- ✅ `.env.example` template
- ✅ Patient data directories ignored
- ✅ Model files excluded
- ✅ API key patterns blocked
- ✅ Security documentation
- ✅ Safe coding practices in source

**But you must:**
- ✅ Never commit `.env` with real keys
- ✅ Use fake data for testing
- ✅ Follow guidelines above
- ✅ Stay vigilant!

---

## 📞 Reporting Security Issues

**Found a security vulnerability?**

**DO:**
- Email privately: [benzaimabdelmouaine@gmail.com](mailto:benzaimabdelmouaine@gmail.com)
- Provide clear description
- Include steps to reproduce
- Allow time for fix

**DON'T:**
- Create public GitHub issue
- Post on social media
- Disclose before fix available

---

## 🎓 For Academic/Research Use

### Thesis Projects

- ✅ Use public, anonymized datasets
- ✅ Cite data sources
- ✅ Document methodology
- ✅ Anonymize results
- ❌ Never publish patient data
- ❌ Never share access credentials

### Sharing Research

**Can share:**
- Source code (without secrets)
- Anonymized results
- Aggregate statistics
- Model architecture
- Documentation

**Cannot share:**
- Raw patient data
- API keys
- Access credentials
- Personal information

---

## Summary

**Three Golden Rules:**

1. **🔑 Never commit secrets** - Use environment variables
2. **🏥 Protect patient data** - Anonymize, encrypt, secure
3. **🔍 Review before commit** - Always check `git status` and `git diff`

**When in doubt:**
- Ask team lead
- Review this guide
- Err on side of caution
- DON'T commit if unsure

---

**Security is everyone's responsibility!** 🔒

Stay safe and secure! If you have questions, review this guide or consult your security team.
