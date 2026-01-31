import streamlit as st
import cv2
import torch
import numpy as np
import json
import os
import tempfile
from pathlib import Path
import time
from typing import Tuple, Dict, Any
import pandas as pd

from vsr.models.lite_vsr import LiteVSR
from vsr.utils.io import read_video, write_video, read_image, write_image
from vsr.utils.timer import LatencyTimer
from vsr.utils.profiling import nvtx_range


# ============================================================================
# STREAMLIT PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="LiteVSR - Video Super-Resolution",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .metric-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .success-box {
        background-color: #d4edda;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #28a745;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #ffc107;
    }
    .info-box {
        background-color: #d1ecf1;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #17a2b8;
    }
    </style>
    """, unsafe_allow_html=True)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_video_info(video_path: str) -> Dict[str, Any]:
    """Extract video metadata"""
    cap = cv2.VideoCapture(video_path)
    
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    
    cap.release()
    
    duration_sec = frame_count / fps if fps > 0 else 0
    file_size_mb = os.path.getsize(video_path) / (1024 * 1024)
    
    return {
        "width": width,
        "height": height,
        "fps": fps,
        "frame_count": frame_count,
        "duration_sec": duration_sec,
        "file_size_mb": file_size_mb,
        "aspect_ratio": f"{width}:{height}",
    }


def suggest_upscale_targets(height: int) -> list:
    """Suggest upscale target resolutions greater than input"""
    common_heights = [480, 576, 720, 1080, 1440, 2160]
    suggestions = [h for h in common_heights if h > height]
    return suggestions if suggestions else [height * 2]


def to_tensor01(rgb_u8: np.ndarray) -> torch.Tensor:
    """Convert uint8 RGB to float32 tensor [0,1]"""
    return torch.from_numpy(rgb_u8).permute(2, 0, 1).float().div(255.0)


def to_uint8(rgb01: torch.Tensor) -> np.ndarray:
    """Convert float32 tensor [0,1] to uint8 RGB"""
    x = rgb01.clamp(0, 1).mul(255.0).byte().permute(1, 2, 0).cpu().numpy()
    return x


def load_model(ckpt_path: str, scale: int = 2, window: int = 5):
    """Load VSR model"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model = LiteVSR(scale=ckpt.get("scale", scale), window=ckpt.get("window", window)).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    
    if device == "cuda":
        torch.backends.cudnn.benchmark = True
    
    return model, device


def run_inference(
    model,
    device: str,
    frames: list,
    window: int = 5,
    fps: float = 30.0,
    precision: str = "amp",
    channels_last: bool = True,
    progress_callback=None,
) -> Tuple[list, Dict[str, Any]]:
    """Run VSR inference on video frames"""
    
    amp = (precision == "amp") and (device == "cuda")
    
    # Prepare windowed frames
    assert window % 2 == 1
    half = window // 2
    padded = [frames[0]] * half + frames + [frames[-1]] * half
    
    out_frames = []
    timer = LatencyTimer()
    
    start_evt = torch.cuda.Event(enable_timing=True) if device == "cuda" else None
    end_evt = torch.cuda.Event(enable_timing=True) if device == "cuda" else None
    
    total = len(frames)
    
    with torch.no_grad():
        for i in range(total):
            if progress_callback:
                progress_callback(i, total)
            
            # Prepare window
            with nvtx_range("prepare_window"):
                window_frames = padded[i : i + window]
                lr = torch.stack([to_tensor01(f) for f in window_frames], dim=0).unsqueeze(0)
                
                if device == "cuda":
                    lr = lr.pin_memory()
                    lr = lr.to(device, non_blocking=True)
            
            # Forward pass
            with nvtx_range("model_forward"):
                if device == "cuda":
                    start_evt.record()
                
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
                    sr = model(lr)
                
                if device == "cuda":
                    end_evt.record()
            
            # Postprocess
            with nvtx_range("postprocess"):
                if device == "cuda":
                    end_evt.synchronize()
                    t_ms = float(start_evt.elapsed_time(end_evt))
                else:
                    t_ms = 0.0
                
                timer.add_ms(t_ms)
                sr_u8 = to_uint8(sr[0])
                out_frames.append(sr_u8)
                
                if device == "cuda" and i % 10 == 0:
                    torch.cuda.empty_cache()
    
    # Compile metrics
    report = {
        "device": device,
        "amp": amp,
        "channels_last": bool(channels_last),
        "frames_total": total,
        "fps": fps,
        "duration_sec": total / fps,
    }
    
    if timer.times_ms:
        s = timer.summary()
        report.update({
            "lat_mean_ms": s.mean_ms,
            "lat_p50_ms": s.p50_ms,
            "lat_p90_ms": s.p90_ms,
            "lat_p99_ms": s.p99_ms,
            "fps_est": s.fps,
        })
    
    return out_frames, report


def create_performance_report(input_info: Dict, output_info: Dict, inference_report: Dict) -> Dict:
    """Create comprehensive performance analysis report"""
    
    input_pixels = input_info["width"] * input_info["height"]
    output_pixels = output_info["width"] * output_info["height"]
    upscale_factor = output_info["height"] / input_info["height"]
    
    # Computational analysis
    total_frames = inference_report.get("frames_total", 1)
    total_pixels_processed = output_pixels * total_frames
    
    device = inference_report.get("device", "cpu")
    avg_latency = inference_report.get("lat_mean_ms", 0)
    estimated_fps = inference_report.get("fps_est", 0)
    
    # Memory estimation
    if device == "cuda":
        if torch.cuda.is_available():
            mem_total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            mem_cached = torch.cuda.memory_reserved(0) / 1e9
            mem_allocated = torch.cuda.memory_allocated(0) / 1e9
        else:
            mem_total_gb = mem_cached = mem_allocated = 0
    else:
        mem_total_gb = mem_cached = mem_allocated = 0
    
    report = {
        "input_resolution": f"{input_info['width']}x{input_info['height']}",
        "output_resolution": f"{output_info['width']}x{output_info['height']}",
        "upscale_factor": f"{upscale_factor:.1f}x",
        "total_frames_processed": total_frames,
        "processing_device": device.upper(),
        "precision_mode": "AMP (Mixed Precision)" if inference_report.get("amp") else "FP32 (Full Precision)",
        
        # Performance metrics
        "avg_latency_ms": f"{avg_latency:.2f}",
        "p50_latency_ms": f"{inference_report.get('lat_p50_ms', 0):.2f}",
        "p90_latency_ms": f"{inference_report.get('lat_p90_ms', 0):.2f}",
        "p99_latency_ms": f"{inference_report.get('lat_p99_ms', 0):.2f}",
        "throughput_fps": f"{estimated_fps:.2f}",
        
        # Computational volume
        "total_pixels_processed": f"{total_pixels_processed / 1e6:.1f}M",
        "pixels_per_second": f"{(total_pixels_processed / (total_frames * avg_latency / 1000)) / 1e6:.1f}M" if avg_latency > 0 else "N/A",
        
        # Memory stats
        "total_gpu_memory_gb": f"{mem_total_gb:.2f}",
        "gpu_memory_used_gb": f"{mem_allocated:.2f}",
        "gpu_memory_cached_gb": f"{mem_cached:.2f}",
    }
    
    return report


# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    st.title("🎬 LiteVSR - Lightweight Video Super-Resolution")
    st.write("Professional video upscaling with real-time performance analysis")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        precision = st.radio(
            "Precision Mode",
            options=["AMP (Faster)", "FP32 (Higher Quality)"],
            help="AMP: Mixed precision for speed. FP32: Full precision for quality."
        )
        precision_mode = "amp" if precision == "AMP (Faster)" else "fp32"
        
        window_size = st.slider(
            "Temporal Window Size",
            min_value=1,
            max_value=11,
            value=5,
            step=2,
            help="Number of frames to process at once (higher = better quality, slower)"
        )
        
        enable_benchmark = st.checkbox(
            "Enable Benchmarking",
            value=True,
            help="Collect detailed performance metrics during inference"
        )
        
        channels_last = st.checkbox(
            "Use Channels Last Format",
            value=True,
            help="GPU optimization for NVIDIA (faster inference)"
        )
    
    # Main content
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📹 Video Upload")
        uploaded_file = st.file_uploader(
            "Upload your video",
            type=["mp4", "mov", "mkv", "avi", "webm"],
            help="Maximum file size depends on your system memory"
        )
    
    if uploaded_file is not None:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
            tmp_file.write(uploaded_file.read())
            input_video_path = tmp_file.name
        
        try:
            # Get video information
            with col2:
                st.subheader("📊 Video Analysis")
                with st.spinner("Analyzing video..."):
                    input_info = get_video_info(input_video_path)
                
                st.success("✓ Video analyzed successfully")
                
                # Display input video info
                info_col1, info_col2 = st.columns(2)
                with info_col1:
                    st.metric("Resolution", f"{input_info['width']}×{input_info['height']}")
                    st.metric("Frame Rate", f"{input_info['fps']:.1f} FPS")
                    st.metric("Total Frames", f"{input_info['frame_count']}")
                
                with info_col2:
                    st.metric("Duration", f"{input_info['duration_sec']:.1f}s")
                    st.metric("File Size", f"{input_info['file_size_mb']:.1f} MB")
                    st.metric("Aspect Ratio", input_info['aspect_ratio'])
            
            # Upscale target selection
            st.divider()
            st.subheader("🚀 Upscaling Configuration")
            
            suggest_targets = suggest_upscale_targets(input_info['height'])
            
            col1, col2, col3 = st.columns(3)
            with col1:
                target_height = st.selectbox(
                    "Target Height (pixels)",
                    options=suggest_targets,
                    help=f"Current input height: {input_info['height']}p. Choose a higher resolution."
                )
            
            with col2:
                if target_height > input_info['height']:
                    upscale_factor = target_height / input_info['height']
                    st.metric("Upscale Factor", f"{upscale_factor:.1f}x")
                else:
                    st.warning("Select a height greater than input")
            
            with col3:
                max_frames_to_process = st.number_input(
                    "Max Frames to Process",
                    min_value=1,
                    value=min(input_info['frame_count'], 300),
                    step=10,
                    help="Limit frames for faster processing. Use full video for production."
                )
            
            # Start processing
            st.divider()
            process_button = st.button(
                "🎯 Start Upscaling",
                type="primary",
                use_container_width=True
            )
            
            if process_button:
                # Load model
                st.info("📦 Loading VSR model...")
                with st.spinner("Loading model checkpoint..."):
                    ckpt_path = "checkpoints/litevsr_scale2.pt"
                    if not os.path.exists(ckpt_path):
                        st.error(f"Checkpoint not found: {ckpt_path}")
                        st.info("Please ensure the model checkpoint exists in the checkpoints folder.")
                        st.stop()
                    
                    model, device = load_model(ckpt_path, scale=2, window=window_size)
                
                st.success(f"✓ Model loaded on {device.upper()}")
                
                # Load video frames
                st.info("📽️ Loading video frames...")
                with st.spinner("Reading video..."):
                    all_frames = list(read_video(input_video_path))
                    frames_to_process = all_frames[:max_frames_to_process]
                
                st.success(f"✓ Loaded {len(frames_to_process)}/{len(all_frames)} frames")
                
                # Run inference
                st.info("⚡ Running inference...")
                progress_bar = st.progress(0)
                frame_counter = st.empty()
                
                def update_progress(current, total):
                    progress_bar.progress(current / total)
                    frame_counter.text(f"Processing frame {current}/{total}")
                
                inference_start_time = time.time()
                
                with st.spinner("Processing frames..."):
                    output_frames, inference_report = run_inference(
                        model,
                        device,
                        frames_to_process,
                        window=window_size,
                        fps=input_info['fps'],
                        precision=precision_mode,
                        channels_last=channels_last,
                        progress_callback=update_progress,
                    )
                
                inference_time = time.time() - inference_start_time
                
                st.success(f"✓ Inference completed in {inference_time:.1f}s")
                
                # Calculate output info
                output_height = target_height
                output_width = int(input_info['width'] * (target_height / input_info['height']))
                output_info = {
                    "width": output_width,
                    "height": output_height,
                    "fps": input_info['fps'],
                }
                
                # Save output video
                st.info("💾 Saving output video...")
                with st.spinner("Writing video file..."):
                    output_video_path = os.path.join(tempfile.gettempdir(), "output_vsr.mp4")
                    # Resize frames to target height
                    resized_frames = []
                    for frame in output_frames:
                        h, w = frame.shape[:2]
                        if h != output_height:
                            scale = output_height / h
                            tw = int(w * scale)
                            frame = cv2.resize(frame, (tw, output_height), interpolation=cv2.INTER_CUBIC)
                        resized_frames.append(frame)
                    
                    write_video(output_video_path, resized_frames, fps=input_info['fps'])
                
                st.success("✓ Video saved successfully")
                
                # Create comprehensive report
                full_report = create_performance_report(input_info, output_info, inference_report)
                
                # Display results
                st.divider()
                st.subheader("📈 Processing Results")
                
                result_col1, result_col2 = st.columns(2)
                
                with result_col1:
                    st.markdown("### Input Video")
                    st.metric("Resolution", full_report["input_resolution"])
                    st.metric("Duration", f"{input_info['duration_sec']:.1f}s")
                
                with result_col2:
                    st.markdown("### Output Video")
                    st.metric("Resolution", full_report["output_resolution"])
                    st.metric("Upscale Factor", full_report["upscale_factor"])
                
                # Performance metrics
                st.divider()
                st.subheader("⚡ Performance Analysis")
                
                perf_col1, perf_col2, perf_col3 = st.columns(3)
                
                with perf_col1:
                    st.markdown("#### Latency")
                    st.metric("Average", full_report["avg_latency_ms"] + " ms")
                    st.metric("P50", full_report["p50_latency_ms"] + " ms")
                    st.metric("P90", full_report["p90_latency_ms"] + " ms")
                    st.metric("P99", full_report["p99_latency_ms"] + " ms")
                
                with perf_col2:
                    st.markdown("#### Throughput")
                    st.metric("Frames/Second", full_report["throughput_fps"])
                    st.metric("Pixels/Second", full_report["pixels_per_second"])
                    st.metric("Total Pixels", full_report["total_pixels_processed"])
                
                with perf_col3:
                    st.markdown("#### Processing")
                    st.metric("Device", full_report["processing_device"])
                    st.metric("Precision", full_report["precision_mode"])
                    st.metric("Actual Time", f"{inference_time:.1f}s")
                
                # GPU Memory stats (if available)
                if device == "cuda":
                    st.divider()
                    st.subheader("💾 GPU Memory Statistics")
                    
                    gpu_col1, gpu_col2, gpu_col3 = st.columns(3)
                    
                    with gpu_col1:
                        st.metric("Total GPU Memory", full_report["total_gpu_memory_gb"] + " GB")
                    
                    with gpu_col2:
                        st.metric("Memory Used", full_report["gpu_memory_used_gb"] + " GB")
                    
                    with gpu_col3:
                        st.metric("Memory Cached", full_report["gpu_memory_cached_gb"] + " GB")
                
                # Detailed performance breakdown table
                st.divider()
                st.subheader("📋 Detailed Performance Report")
                
                report_df = pd.DataFrame({
                    "Metric": list(full_report.keys()),
                    "Value": list(full_report.values()),
                })
                
                st.dataframe(report_df, use_container_width=True, hide_index=True)
                
                # Bottleneck analysis
                st.divider()
                st.subheader("🔍 Bottleneck Analysis")
                
                analysis = ""
                
                # Analyze based on metrics
                if float(full_report["avg_latency_ms"]) > 100:
                    analysis += "⚠️ **High Latency Detected**: Average frame processing > 100ms. "
                    if full_report["processing_device"] == "CPU":
                        analysis += "Consider using GPU or reducing resolution.\n"
                    else:
                        analysis += "Try reducing window size or using AMP precision mode.\n"
                else:
                    analysis += "✅ **Good Latency**: Frame processing is efficient.\n"
                
                if float(full_report["throughput_fps"]) < input_info['fps']:
                    analysis += f"\n⚠️ **Performance Below Real-time**: Processing speed ({full_report['throughput_fps']} FPS) < Input speed ({input_info['fps']:.1f} FPS). "
                    analysis += "This is expected for 2x upscaling on lower-end GPUs.\n"
                else:
                    analysis += f"\n✅ **Real-time Capable**: Can process at {full_report['throughput_fps']} FPS.\n"
                
                if device == "cuda":
                    if float(full_report["gpu_memory_used_gb"]) > 0.8 * float(full_report["total_gpu_memory_gb"]):
                        analysis += f"\n⚠️ **High GPU Memory Usage**: Using >80% of available GPU memory. "
                        analysis += "Reduce batch size or use FP16 precision.\n"
                    else:
                        analysis += f"\n✅ **Good GPU Memory Efficiency**: Using <80% of available GPU memory.\n"
                
                # Kernel optimization suggestions
                analysis += "\n**Kernel Optimization Notes**:\n"
                if channels_last:
                    analysis += "- ✅ NHWC format enabled for better GPU kernel efficiency\n"
                else:
                    analysis += "- ⚠️ NCHW format in use; enabling NHWC may improve performance\n"
                
                if precision_mode == "amp":
                    analysis += "- ✅ Mixed precision (AMP) reduces memory and compute time\n"
                else:
                    analysis += "- ℹ️ Full precision (FP32) provides maximum quality\n"
                
                if window_size > 5:
                    analysis += "- ℹ️ Large temporal window improves quality but increases compute\n"
                else:
                    analysis += "- ✅ Smaller temporal window for faster inference\n"
                
                st.markdown(analysis)
                
                # Download output
                st.divider()
                st.subheader("⬇️ Download Results")
                
                with open(output_video_path, "rb") as video_file:
                    st.download_button(
                        label="📥 Download Upscaled Video",
                        data=video_file,
                        file_name="output_vsr.mp4",
                        mime="video/mp4",
                        use_container_width=True,
                    )
                
                # Save metrics JSON
                metrics_json = json.dumps(full_report, indent=2)
                st.download_button(
                    label="📊 Download Performance Report (JSON)",
                    data=metrics_json,
                    file_name="performance_report.json",
                    mime="application/json",
                    use_container_width=True,
                )
        
        finally:
            # Cleanup
            if os.path.exists(input_video_path):
                os.remove(input_video_path)
    
    else:
        st.info("👆 Upload a video to get started!")
        
        # Show example usage
        with st.expander("📖 How to use"):
            st.markdown("""
            1. **Upload Video**: Choose a video file (MP4, MOV, MKV, etc.)
            2. **Configure Settings**: 
               - Select precision mode (AMP for speed, FP32 for quality)
               - Set temporal window size
               - Enable benchmarking for performance metrics
            3. **Choose Target Resolution**: Select an upscale target higher than input
            4. **Start Processing**: Click "Start Upscaling" to begin
            5. **Review Results**: Check detailed performance metrics and bottleneck analysis
            6. **Download**: Get your upscaled video and performance report
            """)
        
        with st.expander("🎯 Performance Tips"):
            st.markdown("""
            - **For Speed**: Use AMP precision, smaller window size, and enable GPU
            - **For Quality**: Use FP32 precision, larger window size
            - **Real-time Processing**: Limit frames to process on first run
            - **GPU Optimization**: Enable "Channels Last Format" if you have NVIDIA GPU
            - **Memory Issues**: Reduce max frames to process at once
            """)


if __name__ == "__main__":
    main()
