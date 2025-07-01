# backend.py
from fastapi import FastAPI, UploadFile, File, HTTPException, Response
from fastapi.responses import FileResponse, JSONResponse
import cv2
from ultralytics import YOLO
import tempfile
import pytesseract
from collections import defaultdict, Counter
import os
import difflib
import pandas as pd
from typing import List, Dict, Any
from pydantic import BaseModel
import numpy as np
import json

app = FastAPI()

# Load model
model = YOLO(r'C:\\Personal\\Code\\Internship2\\no_plates_detection\\uparrow_2127_images.pt')

# Storage for processing results and frame indexes
processing_results = {}
frame_indexes = {}

class VideoUpload(BaseModel):
    filename: str

class SearchRequest(BaseModel):
    search_term: str
    video_id: str

def preprocess_for_ocr(crop_img):
    gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    scale_percent = 200
    width = int(gray.shape[1] * scale_percent / 100)
    height = int(gray.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized = cv2.resize(gray, dim, interpolation=cv2.INTER_LINEAR)
    return resized

def ocr_on_crop(crop_img):
    preprocessed_img = preprocess_for_ocr(crop_img)
    config = '--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    text = pytesseract.image_to_string(preprocessed_img, config=config)
    return text.strip()

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea) if (boxAArea + boxBArea - interArea) > 0 else 0

def consensus_text(texts):
    if not texts:
        return ""
    max_len = max(len(t) for t in texts)
    consensus_chars = []
    for i in range(max_len):
        chars_at_pos = [t[i] for t in texts if i < len(t)]
        if chars_at_pos:
            most_common_char, _ = Counter(chars_at_pos).most_common(1)[0]
            consensus_chars.append(most_common_char)
    return ''.join(consensus_chars).strip()

def build_frame_index(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_index = {}
    frame_num = 0
    
    while True:
        pos_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
        frame_index[frame_num] = pos_msec
        ret = cap.grab()  # Fast frame skip
        if not ret:
            break
        frame_num += 1
    
    cap.release()
    return frame_index

@app.post("/upload_video/")
async def upload_video(file: UploadFile = File(...)):
    try:
        video_id = str(hash(file.filename))[:10]
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(file.file.read())
        video_path = tfile.name
        
        # Build frame index immediately
        frame_index = build_frame_index(video_path)
        frame_indexes[video_id] = frame_index
        
        processing_results[video_id] = {
            'video_path': video_path,
            'plate_log': pd.DataFrame(columns=['Detected Text', 'Frame Number', 'Bounding Box']),
            'processed': False
        }
        
        return {"video_id": video_id, "filename": file.filename}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/process_video/{video_id}")
async def process_video(video_id: str):
    if video_id not in processing_results:
        raise HTTPException(status_code=404, detail="Video not found")
    
    if processing_results[video_id]['processed']:
        return {"status": "already_processed"}
    
    try:
        video_data = processing_results[video_id]
        video_path = video_data['video_path']
        plate_log = video_data['plate_log']
        
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = int(total_frames / fps)
        
        count = 0
        success = True
        frame_skip = 10
        
        tracked_plates = []
        next_plate_id = 0
        ocr_history = defaultdict(list)
        
        while success:
            success = cap.grab()  # Fast frame grab
            if count % frame_skip == 0 and success:
                ret, frame = cap.retrieve()
                if not ret:
                    continue
                    
                results = model.predict(
                    source=frame,
                    conf=0.35,
                    iou=0.5,
                    augment=True,
                    verbose=False,
                    imgsz=1280
                )[0]

                current_detections = results.boxes.data.cpu().numpy() if len(results.boxes) > 0 else None
                detected_plates_this_frame = []

                if current_detections is not None:
                    det_boxes = [tuple(map(int, det[:4])) for det in current_detections]

                    matches = []
                    unmatched_detections = set(range(len(det_boxes)))
                    unmatched_tracks = set(range(len(tracked_plates)))

                    for det_idx, det_box in enumerate(det_boxes):
                        best_iou = 0
                        best_track_idx = None
                        for track_idx in unmatched_tracks:
                            track_box = tracked_plates[track_idx]['box']
                            iou_score = iou(det_box, track_box)
                            if iou_score > 0.3 and iou_score > best_iou:
                                best_iou = iou_score
                                best_track_idx = track_idx
                        if best_track_idx is not None:
                            matches.append((det_idx, best_track_idx))
                            unmatched_detections.discard(det_idx)
                            unmatched_tracks.discard(best_track_idx)

                    for det_idx, track_idx in matches:
                        det_box = det_boxes[det_idx]
                        tracked_plates[track_idx]['box'] = det_box
                        plate_id = tracked_plates[track_idx]['id']

                        x1, y1, x2, y2 = det_box
                        crop = frame[y1:y2, x1:x2]
                        detected_text = ocr_on_crop(crop) or "N/A"
                        ocr_history[plate_id].append(detected_text)
                        detected_plates_this_frame.append((det_box, plate_id))

                    for det_idx in unmatched_detections:
                        det_box = det_boxes[det_idx]
                        plate_id = next_plate_id
                        next_plate_id += 1
                        tracked_plates.append({'box': det_box, 'id': plate_id})

                        x1, y1, x2, y2 = det_box
                        crop = frame[y1:y2, x1:x2]
                        detected_text = ocr_on_crop(crop) or "N/A"
                        ocr_history[plate_id].append(detected_text)
                        detected_plates_this_frame.append((det_box, plate_id))

                    for box, plate_id in detected_plates_this_frame:
                        x1, y1, x2, y2 = box
                        final_text = consensus_text(ocr_history[plate_id]) or "N/A"

                        new_row = pd.DataFrame({
                            'Detected Text': [final_text],
                            'Frame Number': [count],
                            'Bounding Box': [box]
                        })
                        plate_log = pd.concat([plate_log, new_row], ignore_index=True)

            count += 1

        cap.release()
        
        # Save results
        processing_results[video_id]['plate_log'] = plate_log
        processing_results[video_id]['processed'] = True
        
        # Save to Excel
        excel_path = f"plate_log_{video_id}.xlsx"
        plate_log.to_excel(excel_path, index=False)
        processing_results[video_id]['excel_path'] = excel_path
        
        return {
            "status": "success",
            "video_info": {
                "duration": duration,
                "fps": fps,
                "total_frames": total_frames
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search_plates/")
async def search_plates(request: SearchRequest):
    if request.video_id not in processing_results:
        raise HTTPException(status_code=404, detail="Video not found")
    
    if not processing_results[request.video_id]['processed']:
        raise HTTPException(status_code=400, detail="Video not processed yet")
    
    plate_log = processing_results[request.video_id]['plate_log']
    matches = difflib.get_close_matches(
        request.search_term, 
        plate_log['Detected Text'], 
        n=100, 
        cutoff=0.5
    )
    
    if matches:
        filtered = plate_log[plate_log['Detected Text'].isin(matches)].sort_values(by='Frame Number')
        results = filtered[['Detected Text', 'Frame Number']].to_dict('records')
        return {"matches": results, "count": len(results)}
    else:
        return {"matches": [], "count": 0}

@app.get("/get_frame_with_specific_plate/{video_id}/{frame_number}/{plate_text}")
async def get_frame_with_specific_plate(video_id: str, frame_number: int, plate_text: str):
    if video_id not in processing_results:
        raise HTTPException(status_code=404, detail="Video not found")
    
    video_path = processing_results[video_id]['video_path']
    plate_log = processing_results[video_id]['plate_log']
    
    cap = cv2.VideoCapture(video_path)
    
    # Use frame index for faster seeking
    if video_id in frame_indexes and frame_number in frame_indexes[video_id]:
        cap.set(cv2.CAP_PROP_POS_MSEC, frame_indexes[video_id][frame_number])
    else:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    
    success, frame = cap.read()
    cap.release()
    
    if not success:
        raise HTTPException(status_code=404, detail="Frame not found")
    
    # Get only the specific plate for this frame
    frame_plates = plate_log[
        (plate_log['Frame Number'] == frame_number) & 
        (plate_log['Detected Text'].str.contains(plate_text, case=False, regex=False))
    ]
    
    # Draw bounding boxes only for matching plates
    for _, row in frame_plates.iterrows():
        box = eval(row['Bounding Box']) if isinstance(row['Bounding Box'], str) else row['Bounding Box']
        x1, y1, x2, y2 = box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        text_pos = (x1, y1 - 10 if y1 - 10 > 10 else y1 + 20)
        cv2.putText(frame, row['Detected Text'], text_pos,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Compress image before sending
    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return Response(content=buffer.tobytes(), media_type="image/jpeg")

@app.get("/download_excel/{video_id}")
async def download_excel(video_id: str):
    if video_id not in processing_results:
        raise HTTPException(status_code=404, detail="Video not found")
    
    if not processing_results[video_id]['processed']:
        raise HTTPException(status_code=400, detail="Video not processed yet")
    
    excel_path = processing_results[video_id].get('excel_path')
    if not excel_path or not os.path.exists(excel_path):
        raise HTTPException(status_code=404, detail="Excel file not found")
    
    return FileResponse(
        excel_path,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        filename=f"plate_log_{video_id}.xlsx"
    )