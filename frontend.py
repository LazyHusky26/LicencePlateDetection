# frontend.py
import streamlit as st
import requests
import pandas as pd

# API configuration
API_BASE_URL = "http://localhost:8000"

st.title("🚗 Number Plate Detection from Video")
st.write("Upload a video, and the system will detect number plates from every frame, display OCR consensus text above boxes, and log the details into an Excel sheet.")

# Session state
if 'video_id' not in st.session_state:
    st.session_state.video_id = None
if 'video_processed' not in st.session_state:
    st.session_state.video_processed = False
if 'plate_log' not in st.session_state:
    st.session_state.plate_log = pd.DataFrame(columns=['Detected Text', 'Frame Number', 'Bounding Box'])
if 'uploaded_filename' not in st.session_state:
    st.session_state.uploaded_filename = None

def upload_video(file):
    try:
        files = {'file': (file.name, file.getvalue(), file.type)}
        response = requests.post(f"{API_BASE_URL}/upload_video/", files=files)
        if response.status_code == 200:
            data = response.json()
            st.session_state.video_id = data['video_id']
            st.session_state.uploaded_filename = data['filename']
            st.session_state.video_processed = False
            st.session_state.plate_log = pd.DataFrame(columns=['Detected Text', 'Frame Number', 'Bounding Box'])
            return True
        else:
            st.error(f"Upload failed: {response.json().get('detail', 'Unknown error')}")
            return False
    except Exception as e:
        st.error(f"Upload error: {str(e)}")
        return False

def process_video():
    if not st.session_state.video_id:
        st.error("No video uploaded")
        return False
    
    try:
        with st.spinner("Processing video..."):
            response = requests.post(f"{API_BASE_URL}/process_video/{st.session_state.video_id}")
            if response.status_code == 200:
                data = response.json()
                if data['status'] == 'already_processed':
                    st.info("Video already processed")
                else:
                    st.success("✅ Done processing!")
                st.session_state.video_processed = True
                return True
            else:
                st.error(f"Processing failed: {response.json().get('detail', 'Unknown error')}")
                return False
    except Exception as e:
        st.error(f"Processing error: {str(e)}")
        return False

def search_plates(search_term):
    if not st.session_state.video_id:
        st.error("No video processed")
        return None
    
    try:
        payload = {
            "search_term": search_term,
            "video_id": st.session_state.video_id
        }
        response = requests.post(f"{API_BASE_URL}/search_plates/", json=payload)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Search failed: {response.json().get('detail', 'Unknown error')}")
            return None
    except Exception as e:
        st.error(f"Search error: {str(e)}")
        return None

def get_frame_image(frame_number, plate_text):
    try:
        response = requests.get(
            f"{API_BASE_URL}/get_frame_with_specific_plate/{st.session_state.video_id}/{frame_number}/{plate_text}"
        )
        if response.status_code == 200:
            return response.content
        else:
            st.error(f"Failed to get frame: {response.json().get('detail', 'Unknown error')}")
            return None
    except Exception as e:
        st.error(f"Frame retrieval error: {str(e)}")
        return None

def download_excel():
    try:
        response = requests.get(f"{API_BASE_URL}/download_excel/{st.session_state.video_id}")
        if response.status_code == 200:
            return response.content
        else:
            st.error(f"Download failed: {response.json().get('detail', 'Unknown error')}")
            return None
    except Exception as e:
        st.error(f"Download error: {str(e)}")
        return None

# File uploader
video_file = st.file_uploader(
    "Upload a video", 
    type=["mp4", "avi", "mov"],
    key="video_uploader"
)

if video_file is not None:
    if 'uploaded_filename' not in st.session_state or st.session_state.uploaded_filename != video_file.name:
        # New video uploaded
        if upload_video(video_file):
            st.video(video_file)
    else:
        # Same video uploaded again
        st.info("Same video detected. Click below to reprocess it.")
        if st.button("🔄 Reset & Reprocess", key="reprocess_btn"):
            if upload_video(video_file):
                st.video(video_file)

    # Process video button
    if st.session_state.video_id and not st.session_state.video_processed:
        if st.button("▶️ Process Video", key="process_btn"):
            process_video()

# Download Excel button
if st.session_state.video_processed:
    excel_data = download_excel()
    if excel_data:
        st.download_button(
            label="📥 Download Excel Sheet",
            data=excel_data,
            file_name=f"plate_log_{st.session_state.video_id}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="excel_download_btn"
        )

# Search functionality
search_term = st.text_input(
    "Search for a number plate:",
    key="plate_search_input"
)

if search_term and st.session_state.video_processed:
    search_results = search_plates(search_term)
    if search_results and search_results['count'] > 0:
        st.write(f"Found {search_results['count']} matching plates:")
        df = pd.DataFrame(search_results['matches'])
        
        # Ensure consistent column names
        df = df.rename(columns={
            'Frame Number': 'frame_number',
            'Detected Text': 'detected_text'
        })
        
        st.dataframe(df, use_container_width=True)

        if not df.empty:
            selected_row = st.selectbox(
                "Select a detection to view:",
                options=df.itertuples(),
                format_func=lambda x: f"Frame {x.frame_number}",
                key="plate_selection_box"
            )
            
            if selected_row:
                # Get the image bytes without downloading
                frame_bytes = get_frame_image(selected_row.frame_number, selected_row.detected_text)
                if frame_bytes:
                    # Display the image directly in Streamlit
                    st.subheader(f"📸 Frame {selected_row.frame_number}")
                    st.image(frame_bytes, channels="RGB")
    elif search_results:
        st.warning("No matching plates found.")