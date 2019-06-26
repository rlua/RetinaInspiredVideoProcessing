# RetinaInspiredVideoProcessing

To create an event-driven representation of a video (say `v_Archery_g01_c01.avi`) side-by-side with the original video, run:

```
python PureEDR_retina_convert.py --input_path="example_videos/v_Archery_g01_c01.avi" --out_path="Test_EDR.mov"
```

To create a Reichardt correlator of a video side-by-side with the original video, run:
```
python ReichardtTest_retina_convert.py --input_path="example_videos/v_Archery_g01_c01.avi" --out_path="Test_ReichardtDiagonal2.mov"
```
