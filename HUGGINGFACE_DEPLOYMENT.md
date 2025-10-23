# Hugging Face Spaces Deployment Guide

## 🚀 Quick Deployment

### Step 1: Prepare Your Space

1. Go to [Hugging Face Spaces](https://huggingface.co/spaces)
2. Click "Create new Space"
3. Configure:
   - **Name**: `nsn-integration-dashboard`
   - **License**: MIT
   - **SDK**: Gradio
   - **Hardware**: CPU Basic (free) or GPU for faster inference

### Step 2: Upload Files

Upload these files to your Space:

```
nsn-integration-dashboard/
├── app.py                              # Entry point
├── huggingface_dashboard.py            # Main dashboard
├── backend_telemetry_rank_adapter.py   # Module 1
├── edit_propagation_engine.py          # Module 2
├── rank_feedback_generator.py          # Module 3
├── ensemble_inference_manager.py       # Module 4
├── requirements_dashboard.txt          # Dependencies
└── README.md                           # Space README (use README_SPACES.md)
```

### Step 3: Configure Space

Create or update `README.md` in your Space root with the frontmatter from `README_SPACES.md`:

```yaml
---
title: NSN Integration Dashboard
emoji: 🚀
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.0.0
app_file: app.py
pinned: false
license: mit
---
```

### Step 4: Deploy

1. Commit and push files to your Space
2. Hugging Face will automatically build and deploy
3. Wait 2-3 minutes for build to complete
4. Your dashboard will be live at: `https://huggingface.co/spaces/nurcholish/nsn-integration-dashboard`

---

## 📁 File Structure

### Required Files

#### `app.py` (Entry Point)
```python
from huggingface_dashboard import create_gradio_interface

demo = create_gradio_interface()

if __name__ == '__main__':
    demo.launch(server_name="0.0.0.0", server_port=7860)
```

#### `requirements_dashboard.txt` (Dependencies)
```
numpy>=1.21.0
pandas>=1.3.0
gradio>=4.0.0
plotly>=5.14.0
python-dateutil>=2.8.2
```

#### Module Files
- `backend_telemetry_rank_adapter.py`
- `edit_propagation_engine.py`
- `rank_feedback_generator.py`
- `ensemble_inference_manager.py`
- `huggingface_dashboard.py`

---

## 🎨 Customization

### Branding

Edit `huggingface_dashboard.py` to customize:

```python
with gr.Blocks(title="Your Custom Title", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🚀 Your Custom Header
    Your custom description
    """)
```

### Themes

Available Gradio themes:
- `gr.themes.Soft()` (default)
- `gr.themes.Base()`
- `gr.themes.Glass()`
- `gr.themes.Monochrome()`

### Colors

Customize in Space README frontmatter:
```yaml
colorFrom: blue  # Start color
colorTo: purple  # End color
```

---

## 🔧 Advanced Configuration

### Enable GPU

For faster inference, upgrade to GPU hardware:

1. Go to Space Settings
2. Select "Hardware" tab
3. Choose GPU tier (T4, A10G, or A100)
4. Confirm upgrade

### Add Authentication

Restrict access to your Space:

```python
demo.launch(
    auth=("username", "password"),
    server_name="0.0.0.0",
    server_port=7860
)
```

### Enable Queue

For high traffic:

```python
demo.queue(concurrency_count=3)
demo.launch()
```

### Add Analytics

Track usage with Gradio Analytics:

```python
demo.launch(
    analytics_enabled=True,
    server_name="0.0.0.0"
)
```

---

## 📊 Monitoring

### View Logs

1. Go to your Space page
2. Click "Logs" tab
3. Monitor real-time activity

### Check Status

Space status indicators:
- 🟢 **Running**: Space is live
- 🟡 **Building**: Deployment in progress
- 🔴 **Error**: Build failed (check logs)
- ⚪ **Sleeping**: Inactive (will wake on access)

---

## 🐛 Troubleshooting

### Build Fails

**Issue**: Dependencies not installing

**Solution**: Check `requirements_dashboard.txt` syntax
```bash
# Test locally first
pip install -r requirements_dashboard.txt
```

### Import Errors

**Issue**: Module not found

**Solution**: Ensure all module files are uploaded and paths are correct
```python
# Use relative imports
from backend_telemetry_rank_adapter import BackendTelemetryRankAdapter
```

### Memory Issues

**Issue**: Out of memory errors

**Solution**: 
1. Upgrade to larger hardware tier
2. Reduce batch sizes in visualizations
3. Optimize data structures

### Slow Performance

**Issue**: Dashboard is slow

**Solution**:
1. Enable caching: `@gr.cache()`
2. Reduce plot complexity
3. Upgrade to GPU hardware

---

## 🔄 Updates

### Update Your Space

1. Edit files locally
2. Test changes: `python app.py`
3. Push to Space repository
4. Space will auto-rebuild

### Version Control

Use Git for version control:

```bash
# Clone your Space
git clone https://huggingface.co/spaces/nurcholish/nsn-integration-dashboard

# Make changes
git add .
git commit -m "Update dashboard"
git push
```

---

## 🌐 Sharing

### Public Access

Share your Space URL:
```
https://huggingface.co/spaces/nurcholish/nsn-integration-dashboard
```

### Embed in Website

```html
<iframe
  src="https://nurcholish-nsn-integration-dashboard.hf.space"
  frameborder="0"
  width="100%"
  height="800"
></iframe>
```

### API Access

Use Gradio Client:

```python
from gradio_client import Client

client = Client("nurcholish/nsn-integration-dashboard")
result = client.predict("input_data", api_name="/predict")
```

---

## 📈 Scaling

### Handle High Traffic

1. **Enable Queue**: `demo.queue()`
2. **Upgrade Hardware**: Use GPU or larger CPU
3. **Optimize Code**: Cache results, reduce computations
4. **Use CDN**: For static assets

### Multiple Replicas

For enterprise use, contact Hugging Face for:
- Dedicated hardware
- Multiple replicas
- Custom domains
- SLA guarantees

---

## 💰 Costs

### Free Tier
- CPU Basic: Free
- 2 vCPU, 16GB RAM
- Sleeps after 48h inactivity

### Paid Tiers
- **CPU Upgrade**: $0.03/hour
- **T4 GPU**: $0.60/hour
- **A10G GPU**: $1.05/hour
- **A100 GPU**: $3.15/hour

---

## 📚 Resources

- [Gradio Documentation](https://gradio.app/docs/)
- [Hugging Face Spaces Guide](https://huggingface.co/docs/hub/spaces)
- [Gradio Themes](https://gradio.app/theming-guide/)
- [Example Spaces](https://huggingface.co/spaces)

---

## 🤝 Support

Need help?
- [Hugging Face Forums](https://discuss.huggingface.co/)
- [Gradio Discord](https://discord.gg/gradio)
- [GitHub Issues](https://github.com/your-repo/quantum-limit-graph/issues)

---

## ✅ Deployment Checklist

- [ ] Create Hugging Face account
- [ ] Create new Space
- [ ] Upload all required files
- [ ] Configure README with frontmatter
- [ ] Test locally before deploying
- [ ] Monitor build logs
- [ ] Test all dashboard panels
- [ ] Share Space URL
- [ ] Add to documentation
- [ ] Announce to community

---

**Your NSN Integration Dashboard is now live! 🎉**

Share it with the community and start collecting contributions!
