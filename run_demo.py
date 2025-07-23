#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
VEO3 Directors - Demo Version (UI Only)
Simplified version for testing UI functionality without model dependencies
"""

import os
import json
import random
import logging
from pathlib import Path
from datetime import datetime

try:
    import gradio as gr
    import pandas as pd
    from loguru import logger
    import requests
except ImportError as e:
    print(f"Missing required packages. Please install: {e}")
    print("Try: pip install gradio pandas loguru requests")
    exit(1)

# Environment settings
DEMO_MODE = True
logger.info("Running in DEMO MODE - No actual AI models loaded")

# Mock data
DEFAULT_TOPICS_KO = [
    "시간 여행자의 마지막 선택",
    "AI가 사랑에 빠진 날", 
    "잊혀진 도서관의 비밀",
    "평행우주의 또 다른 나",
    "마지막 인류의 일기"
]

DEFAULT_STARTERS_KO = [
    "그날 아침, 하늘에서 시계가 떨어졌다.",
    "커피잔에 비친 내 얼굴이 낯설었다.", 
    "도서관 13번 서가는 항상 비어있었다.",
    "전화벨이 울렸다. 30년 전에 죽은 아버지였다.",
    "거울 속 나는 웃고 있지 않았다."
]

DEFAULT_TOPICS_EN = [
    "The Time Traveler's Final Choice",
    "The Day AI Fell in Love",
    "Secret of the Forgotten Library", 
    "Another Me in a Parallel Universe",
    "Diary of the Last Human"
]

DEFAULT_STARTERS_EN = [
    "That morning, a clock fell from the sky.",
    "My reflection in the coffee cup looked unfamiliar.",
    "Shelf 13 in the library was always empty.", 
    "The phone rang. It was my father who died 30 years ago.",
    "The me in the mirror wasn't smiling."
]

# Load JSON data
def load_json_safe(path: str, default_data: list) -> list:
    try:
        p = Path(path)
        if not p.is_file():
            return default_data
        with p.open(encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default_data

TOPICS_KO = load_json_safe("story.json", DEFAULT_TOPICS_KO)
STARTERS_KO = load_json_safe("first.json", DEFAULT_STARTERS_KO)
TOPICS_EN = load_json_safe("story_en.json", DEFAULT_TOPICS_EN)
STARTERS_EN = load_json_safe("first_en.json", DEFAULT_STARTERS_EN)

TOPIC_DICT_KO = {"Genre": TOPICS_KO}
TOPIC_DICT_EN = {"Genre": TOPICS_EN}
CATEGORY_LIST = ["Genre"]

# Mock functions
def pick_seed_global(category: str, use_korean: bool) -> dict:
    topic_dict = TOPIC_DICT_KO if use_korean else TOPIC_DICT_EN
    starters = STARTERS_KO if use_korean else STARTERS_EN
    
    if category == "Random":
        pool = [s for lst in topic_dict.values() for s in lst]
    else:
        pool = topic_dict.get(category, [])
        if not pool:
            pool = [s for lst in topic_dict.values() for s in lst]
    
    topic = random.choice(pool)
    topic = topic.split(" (")[0] if " (" in topic else topic
    opening = random.choice(starters)
    return {"카테고리": category, "소재": topic, "첫 문장": opening}

def demo_chat_response(message, history, max_tokens=1000, use_korean=False, system_prompt=""):
    """Mock chat response for demo"""
    user_msg = message.get("text", "") if isinstance(message, dict) else str(message)
    
    if "continued" in user_msg.lower() or "이어서" in user_msg or "계속" in user_msg:
        return """AI💘: 

In the depths of the hidden control room beneath the old library, the middle-aged librarian stands frozen before a wall of glowing monitors as the camera executes a slow [dolly in] toward shelf 13, then transitions to a dramatic [crane down] following him into the secret passage bathed in shifting amber library lights that gradually transform into cold blue technological glow, his trembling hands clutching an ancient leather-bound book while his wire-rimmed glasses reflect the screens displaying "KNOWLEDGE IS POWER" in bold white letters across the central monitor, the 24mm wide-angle lens capturing his nervous anticipation shifting to awe as mechanical whirs and digital hums fill the 8-second sequence.

계속 또는 이어서라고 입력하시면 다음 영상 프롬프트를 생성하겠습니다."""
    
    return f"""AI💘:

DEMO MODE: This is a mock response to demonstrate the UI functionality.

Your input: {user_msg[:100]}{'...' if len(user_msg) > 100 else ''}

In a real deployment, this would generate a detailed video prompt based on your story seed using AI models.

계속 또는 이어서라고 입력하시면 다음 영상 프롬프트를 생성하겠습니다."""

def mock_video_generation(prompt, neg_prompt, scale, height=480, width=832, duration=4, steps=4, seed=2025, randomize_seed=False, enable_audio=True, audio_neg="", audio_steps=25, audio_cfg=4.5):
    """Mock video generation for demo"""
    current_seed = random.randint(0, 999999) if randomize_seed else seed
    logger.info(f"DEMO: Mock video generation - {width}x{height}, {duration}s, {steps} steps, seed: {current_seed}")
    logger.info(f"DEMO: Prompt: {prompt[:100]}...")
    
    # Return None for video (no actual video generated) and the seed
    return None, current_seed

# CSS styling
css = """
.gradio-container {
    max-width: 1800px !important;
    margin: 0 auto !important;
    font-family: 'Pretendard', -apple-system, BlinkMacSystemFont, system-ui, sans-serif !important;
}

.main-header {
    text-align: center;
    padding: 2rem 0;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 15px;
    margin-bottom: 2rem;
    box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
}

.main-header h1 {
    color: white !important;
    font-size: 3rem !important;
    font-weight: 800 !important;
    margin: 0 !important;
    text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.2);
}

.demo-warning {
    background: #fef3c7;
    border: 2px solid #f59e0b;
    border-radius: 10px;
    padding: 1rem;
    margin: 1rem 0;
    text-align: center;
    color: #92400e;
    font-weight: bold;
}
"""

# Create Gradio interface
with gr.Blocks(css=css, theme=gr.themes.Soft(), title="VEO3 Directors - Demo") as demo:
    # Header
    gr.HTML("""
        <div class="main-header">
            <h1>🎬 VEO3 Directors - Demo</h1>
            <p>Complete Video Creation Suite: Story → Script → Video + Audio</p>
        </div>
    """)
    
    # Demo warning
    gr.HTML("""
        <div class="demo-warning">
            ⚠️ DEMO MODE: AI models are disabled. This demo shows UI functionality only.
            <br>For full functionality, you need to configure API keys and install model dependencies.
        </div>
    """)
    
    with gr.Tabs():
        # Tab 1: Story & Script Generation
        with gr.TabItem("📝 Story & Script Generation"):
            gr.Markdown("### 🎲 Step 1: Generate Story Seed")
            
            with gr.Row():
                with gr.Column(scale=3):
                    category_dd = gr.Dropdown(
                        label="Seed Category",
                        choices=["Random"] + CATEGORY_LIST,
                        value="Random",
                        interactive=True
                    )
                with gr.Column(scale=1):
                    use_korean = gr.Checkbox(label="🇰🇷 Korean", value=False)
            
            seed_display = gr.Textbox(
                label="Generated Story Seed",
                lines=4,
                interactive=False,
                placeholder="Click 'Generate Story Seed' to create a new story seed..."
            )
            
            generate_seed_btn = gr.Button("🎲 Generate Story Seed", variant="primary")
            
            gr.Markdown("### 🎥 Step 2: Generate Video Script & Prompt")
            
            prompt_chat = gr.ChatInterface(
                fn=demo_chat_response,
                type="messages",
                chatbot=gr.Chatbot(type="messages", height=400),
                textbox=gr.MultimodalTextbox(
                    file_types=[],
                    placeholder="Enter topic and first sentence to generate video prompt...",
                    lines=3
                ),
                additional_inputs=[gr.Slider(100, 8000, 1000, label="Max Tokens"), use_korean],
                examples=[
                    [{"text": "continued...", "files": []}],
                    [{"text": "story random generation", "files": []}],
                    [{"text": "이어서 계속", "files": []}]
                ]
            )
        
        # Tab 2: Video Generation
        with gr.TabItem("🎬 Video Generation"):
            gr.Markdown("### 🎥 Step 3: Generate Video with Audio")
            
            with gr.Row():
                with gr.Column(scale=1):
                    video_prompt = gr.Textbox(
                        label="✨ Video Prompt",
                        placeholder="Paste your generated prompt here or write your own...",
                        lines=4
                    )
                    
                    with gr.Row():
                        duration_seconds = gr.Slider(1, 8, 4, step=1, label="Duration (seconds)")
                        steps = gr.Slider(1, 8, 4, step=1, label="Inference Steps")
                    
                    with gr.Row():
                        height = gr.Slider(128, 896, 480, step=32, label="Height")
                        width = gr.Slider(128, 896, 832, step=32, label="Width")
                    
                    with gr.Row():
                        seed = gr.Slider(0, 999999, 2025, step=1, label="Seed")
                        randomize_seed = gr.Checkbox(label="Random Seed", value=True)
                    
                    enable_audio = gr.Checkbox(label="🔊 Enable Audio", value=True)
                    
                    generate_video_btn = gr.Button("🎬 Generate Video (Demo)", variant="primary")
                
                with gr.Column(scale=1):
                    video_output = gr.Video(label="Generated Video (Demo Mode - No Output)")
                    
                    gr.HTML("""
                        <div style="background: #e5e7eb; padding: 1rem; border-radius: 8px; margin-top: 1rem;">
                            <p><strong>💡 Demo Mode Info:</strong></p>
                            <ul>
                                <li>✅ UI is fully functional</li>
                                <li>✅ Story seed generation works</li>
                                <li>✅ Chat interface responds</li>
                                <li>❌ No actual video/audio generation</li>
                                <li>❌ AI models are disabled</li>
                            </ul>
                        </div>
                    """)
    
    # Event handlers
    def generate_seed_display(category, use_korean):
        seed = pick_seed_global(category, use_korean)
        if use_korean:
            txt = f"🎲 카테고리: {seed['카테고리']}\n🎭 주제: {seed['소재']}\n🏁 첫 문장: {seed['첫 문장']}"
        else:
            txt = f"🎲 CATEGORY: {seed['카테고리']}\n🎭 TOPIC: {seed['소재']}\n🏁 FIRST LINE: {seed['첫 문장']}"
        return txt
    
    generate_seed_btn.click(
        fn=generate_seed_display,
        inputs=[category_dd, use_korean],
        outputs=[seed_display]
    )
    
    generate_video_btn.click(
        fn=mock_video_generation,
        inputs=[video_prompt, gr.Textbox(visible=False), gr.Slider(visible=False), 
                height, width, duration_seconds, steps, seed, randomize_seed,
                enable_audio, gr.Textbox(visible=False), gr.Slider(visible=False), gr.Slider(visible=False)],
        outputs=[video_output, seed]
    )

if __name__ == "__main__":
    print("Starting VEO3 Directors Demo...")
    print("Demo Mode: UI functionality only")
    print("Access the interface at: http://localhost:7861")
    
    try:
        demo.launch(
            server_name="127.0.0.1",
            server_port=7861,
            share=False,
            debug=False
        )
    except Exception as e:
        print(f"Failed to launch: {e}")
        print("Try installing missing dependencies:")
        print("pip install gradio pandas loguru requests")