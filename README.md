# VEO3 Directors - AI Video Creation Suite

A complete video creation suite that combines story generation, script writing, and video/audio generation using advanced AI models.

## 🚀 Features

- **📝 Story Seed Generation**: Generate creative story seeds with customizable categories
- **🎥 AI Script Generation**: Convert stories into detailed video production prompts
- **🎬 Video Generation**: Create videos using Wan2.1-T2V-14B model with NAG technology
- **🎵 Audio Generation**: Automatic audio generation and synchronization using MMAudio
- **🌐 Web Interface**: Beautiful, responsive Gradio-based web interface

## 🏃‍♂️ Quick Start

### Demo Mode (Recommended for Testing)

For testing the UI without downloading large AI models:

```bash
# Clone the repository
git clone https://github.com/AI-Adventist/veo3-WebUI.git
cd veo3-WebUI

# Install basic dependencies
pip install gradio pandas loguru requests

# Run demo version
python run_demo.py
```

Open http://localhost:7860 in your browser.

### Full Version (Requires GPU and Model Downloads)

⚠️ **Requirements**: 24GB+ GPU memory, ~50GB storage space

```bash
# Install all dependencies
pip install -r requirements.txt

# Set environment variables
export FRIENDLI_TOKEN="your_friendli_api_token"
export SERPHOUSE_API_KEY="your_serphouse_api_key"  # Optional

# Run full version
python app.py
```

## 📁 Project Structure

```
VEO3-Directors/
├── app.py                 # Main application (full version)
├── run_demo.py           # Demo version (UI only)
├── requirements.txt      # Python dependencies
├── src/                  # Custom pipeline implementations
│   ├── pipeline_wan_nag.py
│   ├── transformer_wan_nag.py
│   └── attention_wan_nag.py
├── mmaudio/             # Audio generation module
├── docs/                # Documentation and assets
├── story.json           # Korean story seeds
├── story_en.json        # English story seeds
├── first.json           # Korean story starters
└── first_en.json        # English story starters
```

## 🎯 Usage Workflow

1. **Generate Story Seed**: Use the story generator to create creative prompts
2. **Create Script**: Convert your story into detailed video production prompts using AI
3. **Generate Video**: Create videos with automatic audio synchronization

## 🔧 Configuration

### Environment Variables

- `FRIENDLI_TOKEN`: Required for AI script generation
- `SERPHOUSE_API_KEY`: Optional for enhanced search capabilities

### Model Settings

- **Video Model**: Wan-AI/Wan2.1-T2V-14B-Diffusers
- **Audio Model**: MMAudio large_44k_v2
- **Device**: CUDA (GPU required for full version)

## 🎨 Features Details

### NAG Technology
- Uses Negative-Augmented Generation for better video quality
- Configurable guidance scale and steps
- Support for custom negative prompts

### Audio Generation
- Automatic audio generation based on video content
- Configurable audio guidance strength
- Support for custom audio negative prompts

### Multi-language Support
- Korean and English interface
- Localized story seeds and prompts

## 🔍 Demo vs Full Version

| Feature | Demo Mode | Full Version |
|---------|-----------|--------------|
| Story Generation | ✅ | ✅ |
| UI Interface | ✅ | ✅ |
| Chat Interface | ✅ (Mock) | ✅ (AI) |
| Video Generation | ❌ | ✅ |
| Audio Generation | ❌ | ✅ |
| GPU Required | ❌ | ✅ |
| Model Downloads | ❌ | ✅ (~50GB) |

## 🐛 Troubleshooting

### Common Issues

1. **"No module named 'mmaudio'"**: Run `pip install -e .` in the project directory
2. **CUDA out of memory**: Reduce video resolution or duration
3. **API errors**: Check your FRIENDLI_TOKEN configuration

### Demo Mode Issues

If the demo doesn't start:
```bash
pip install gradio pandas loguru requests python-dotenv
python run_demo.py
```

## 📄 License

This project is based on the original VEO3-Directors from HuggingFace Spaces.

## 🤝 Contributing

Feel free to submit issues and pull requests to improve the project.

## 🔗 Links

- [Original HuggingFace Space](https://huggingface.co/spaces/ginigen/VEO3-Directors)
- [Wan2.1-T2V Model](https://huggingface.co/Wan-AI/Wan2.1-T2V-14B-Diffusers)
- [MMAudio](https://github.com/hkchengrex/MMAudio)

---

**Note**: This is a demo/development version. For production use, ensure proper GPU resources and API configurations.