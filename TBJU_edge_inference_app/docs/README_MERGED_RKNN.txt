Merged YOLO + TBJU OCR RKNN deploy package
==========================================

Files:

  inference_tbju.py          Board-side static image inference script
  inference_tbju_merged.py   Same as inference_tbju.py, kept as explicit backup
  inference_tbju_stream.py   Video file and realtime camera demo script
  merged_yolov8.rknn         Default merged YOLO FP model
  merged_yolov8_fp.rknn      Merged YOLO FP model
  merged_yolov8_i8.rknn      Merged YOLO INT8 model
  merged_classes.txt         Class id mapping for merged YOLO
  rec_tbju.rknn              OCR FP model
  rec_tbju_fp.rknn           OCR FP model backup
  ppocr_keys_v1.txt          OCR CTC character dictionary

Merged YOLO classes:

  0 TBJU_region
  1 carriage_rim_region
  2 carriage_rim_debris
  3 track_region
  4 track_intrusion_debris
  5 door_region

Pipeline:

  1. Run merged YOLO once on the whole image.
  2. Draw all six detection classes.
  3. Only class 0, TBJU_region, is cropped and sent to OCR.
  4. OCR input is NCHW, so inference_tbju.py keeps data_format='nchw'.
  5. Other classes are detection-only and are not sent to OCR.

Board commands:

  cd /home/elf/RKNN_deploy
  python3 inference_tbju.py --image test.jpg --all_cores

Batch:

  python3 inference_tbju.py --image_dir test_images --save_dir results_merged --all_cores

Use INT8 YOLO for comparison:

  python3 inference_tbju.py --image test.jpg --yolo_model merged_yolov8_i8.rknn --all_cores

Detection-only test:

  python3 inference_tbju.py --image test.jpg --skip_ocr --all_cores

Video file demo:

  python3 inference_tbju_stream.py --video demo.mp4 --output results_video/demo_result.avi --all_cores

Video file demo with window:

  export DISPLAY=:0.0
  python3 inference_tbju_stream.py --video demo.mp4 --output results_video/demo_result.avi --show --all_cores

OV13855 camera realtime demo:

  export DISPLAY=:0.0
  python3 inference_tbju_stream.py --camera /dev/video11 --show --fps 30 --all_cores

OV13855 camera realtime demo with recording:

  python3 inference_tbju_stream.py --camera /dev/video11 --show --output results_camera/camera_demo.avi --fps 30 --all_cores

GStreamer camera fallback:

  python3 inference_tbju_stream.py --camera /dev/video11 --gst --show --all_cores

GStreamer camera fallback with recording:

  python3 inference_tbju_stream.py --camera /dev/video11 --gst --show --output results_camera/camera_demo.avi --all_cores
