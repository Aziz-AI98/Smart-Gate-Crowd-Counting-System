import cv2
import gradio as gr
import numpy as np
from ultralytics import YOLO

# 1. تحميل نموذج YOLO نسخة ال Pytorch
model = YOLO("yolov8n.pt")
# استدعاء الدالة 
def process_gate_video(video_path):
   
    if video_path is None:
        return None, "الرجاء رفع ملف فيديو صالح أولاً."

    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # إعداد ملف الفيديو الناتج مع ترميز يدعم تشغيله داخل المتصفح
    output_path = "output_counted_gate.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'avc1') # أو استخدم 'mp4v' حسب المتوفر
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    people_count = 0
    tracked_ids = set()
    line_y = int(height * 0.6) # تحديد مكان خط العد افتراضياً في أسفل/منتصف الفيديو

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # استخدام التتبع (track) وتحديد الفئة 0 الخاصة بالأشخاص (Persons)
        results = model.track(frame, persist=True, classes=[0], verbose=False)

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().numpy()

            for box, track_id in zip(boxes, track_ids):
                x1, y1, x2, y2 = map(int, box)
                center_x = int((x1 + x2) / 2)
                center_y = int(y2)

                # رسم المربع المحيط بالشخص واسم المعرف الخاص به
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # منطق العد عند عبور الخط الافتراضي
                if line_y - 10 < center_y < line_y + 10:
                    if track_id not in tracked_ids:
                        people_count += 1
                        tracked_ids.add(track_id)

        # رسم خط العد باللون الأحمر على الفريم
        cv2.line(frame, (0, line_y), (width, line_y), (0, 0, 255), 3)
        
        # كتابة العداد الحالي على الشاشة
        cv2.putText(frame, f"Total Count: {people_count}", (30, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3)

        out.write(frame)

    cap.release()
    out.release()
    
    result_message = f"تم الانتهاء من المعالجة بنجاح. إجمالي الأشخاص العابرين: {people_count}"
    return output_path, result_message

# 2. بناء واجهة Gradio التفاعلية
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🚪 نظام عد الأشخاص الذكي عبر البوابات")
    gr.Markdown("قم برفع فيديو لكاميرا المراقبة الخاصة بالبوابة، وسيقوم النظام بتعقب الأشخاص وعدهم تلقائياً باستخدام **YOLO** و **OpenCV**.")
    
    with gr.Row():
        with gr.Column():
            video_input = gr.Video(label="فيديو البوابة المدخل")
            submit_btn = gr.Button("بدء العد والمعالجة", variant="primary")
        
        with gr.Column():
            video_output = gr.Video(label="الفيديو بعد المعالجة والعد")
            text_output = gr.Textbox(label="ملخص النتيجة")

    submit_btn.click(
        fn=process_gate_video,
        inputs=video_input,
        outputs=[video_output, text_output]
    )

# تشغيل التطبيق
if __name__ == "__main__":
    demo.launch()
