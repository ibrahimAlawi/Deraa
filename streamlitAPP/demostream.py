import streamlit as st
from diffusers import StableDiffusionPipeline
import matplotlib.pyplot as plt
from transformers import pipeline as hf_pipeline
import sentencepiece  # Ensure sentencepiece is installed
import cv2
from ultralytics import YOLO
from pytube import YouTube
import tempfile
import os

@st.cache_resource
def load_models():
    classifier = hf_pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    translator = hf_pipeline("translation_ar_to_en", model="Helsinki-NLP/opus-mt-ar-en")
    pipeline = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-base")
    return classifier, translator, pipeline

classifier, translator, pipeline = load_models()
labels = ["NSFW", "Safe", "sexy", "religion", "women", "fight", "famous person"]

def classify_sentence(prompt):
    result = classifier(prompt, candidate_labels=labels)
    for label, score in zip(result['labels'], result['scores']):
        if label in ["NSFW", "sexy", "religion", "women", "fight", "famous person"] and score > 0.5:
            return True
    return False

def translate_text(prompt):
    translated_text = translator(prompt)[0]['translation_text']
    return translated_text

def image_generation():
    st.title("Image Generation🖼️")
    prompt = st.text_input("Enter a text:")
    if prompt:
        if any('\u0600' <= char <= '\u06FF' for char in prompt):
            prompt = translate_text(prompt)
        if classify_sentence(prompt):
            st.error("This content is not allowed.")
        else:
            with st.spinner("Generating image..."):
                image = pipeline(prompt=prompt).images[0]
                st.image(image, caption=prompt, use_column_width=True)

def video_Blurring():
    st.title("Video Blurring🎥")
    youtube_url = st.text_input("Enter YouTube video URL:", "https://www.youtube.com/watch?v=EFs9iXd33e4&t=3s")
    if st.button("Process Video"):
        with st.spinner("Downloading and processing video..."):
            yt = YouTube(youtube_url)
            stream = yt.streams.filter(file_extension='mp4').first()
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            stream.download(filename=temp_file.name)
            video_path = temp_file.name
            
            model = YOLO("newbestv3.pt")
            cap = cv2.VideoCapture(video_path)
            assert cap.isOpened(), "Error reading video file"
            original_w, original_h, original_fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
            desired_fps = original_fps
            blur_kernel_size = 1703
            expansion_margin = 70

            
            temp_dir = tempfile.TemporaryDirectory()
            frame_dir = temp_dir.name

            frame_count = 0
            while cap.isOpened():
                success, im0 = cap.read()
                if not success:
                    st.write("Video frame is empty or video processing has been successfully completed.")
                    break
                results = model.predict(im0, show=False)
                boxes = results[0].boxes.xyxy.cpu().tolist()
                clss = results[0].boxes.cls.cpu().tolist()
                if boxes:
                    for box, cls in zip(boxes, clss):
                        x1 = max(0, int(box[0]) - expansion_margin)
                        y1 = max(0, int(box[1]) - expansion_margin)
                        x2 = min(original_w, int(box[2]) + expansion_margin)
                        y2 = min(original_h, int(box[3]) + expansion_margin)
                        obj = im0[y1:y2, x1:x2]
                        blur_obj = cv2.blur(obj, (blur_kernel_size, blur_kernel_size))
                        im0[y1:y2, x1:x2] = blur_obj
                    st.image(im0, channels="BGR")

               
                frame_filename = os.path.join(frame_dir, f"frame_{frame_count:04d}.jpg")
                cv2.imwrite(frame_filename, im0)
                frame_count += 1
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            cap.release()
            cv2.destroyAllWindows()

            
            output_video_path = "processed_video.mp4"
            frame_files = [os.path.join(frame_dir, f) for f in sorted(os.listdir(frame_dir)) if f.endswith(".jpg")]
            frame = cv2.imread(frame_files[0])
            height, width, layers = frame.shape
            video_writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*"mp4v"), desired_fps, (width, height))
            for frame_file in frame_files:
                frame = cv2.imread(frame_file)
                video_writer.write(frame)
            video_writer.release()
            
            temp_dir.cleanup()

            st.video(output_video_path)

def main():
    st.sidebar.title("درع")
    page = st.sidebar.radio("Go to ...", ["Home🏠", "Image Generation🖼️", "Video Blurring🎥"])

    if page == "Home🏠":
        st.markdown(
            """
            <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; height: 80vh;">
                <h1>درع</h1>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.image('logo2.png', use_column_width=True)
        st.write("This application is designed to generate images and apply blurring to video clips that contain scenes of violence or inappropriate content.")
    elif page == "Image Generation🖼️":
        image_generation()
    elif page == "Video Blurring🎥":
        video_Blurring()
        

if __name__ == '__main__':
    main()