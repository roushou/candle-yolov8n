use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::{Module, VarBuilder};
use candle_transformers::object_detection::{Bbox, KeyPoint, non_maximum_suppression};
use model::{Multiples, YoloV8};
use opencv::{
    core::{Mat, MatTraitConst, MatTraitConstManual, Point, Rect, Scalar},
    highgui,
    imgproc::{self, FONT_HERSHEY_SIMPLEX, LINE_8, LINE_AA, put_text, rectangle},
    videoio::{
        CAP_ANY, CAP_PROP_CONVERT_RGB, VideoCapture, VideoCaptureTrait, VideoCaptureTraitConst,
    },
};

mod classes;
mod model;

pub trait Task: Module + Sized {
    fn load(vb: VarBuilder, multiples: Multiples) -> candle_core::Result<Self>;
    fn report(
        pred: &Tensor,
        img: &Mat,
        w: usize,
        h: usize,
        confidence_threshold: f32,
        nms_threshold: f32,
        legend_size: u32,
    ) -> candle_core::Result<Mat>;
}

impl Task for YoloV8 {
    fn load(vb: VarBuilder, multiples: Multiples) -> candle_core::Result<Self> {
        YoloV8::load(vb, multiples, /* num_classes=*/ 80)
    }

    fn report(
        pred: &Tensor,
        img: &Mat,
        w: usize,
        h: usize,
        confidence_threshold: f32,
        nms_threshold: f32,
        legend_size: u32,
    ) -> candle_core::Result<Mat> {
        report_detect(
            pred,
            img,
            w,
            h,
            confidence_threshold,
            nms_threshold,
            legend_size,
        )
    }
}

fn report_detect(
    pred: &Tensor,
    img: &Mat,
    w: usize,
    h: usize,
    confidence_threshold: f32,
    nms_threshold: f32,
    legend_size: u32,
) -> candle_core::Result<Mat> {
    let pred = pred.to_device(&Device::Cpu)?;
    let (pred_size, npreds) = pred.dims2()?;
    let nclasses = pred_size - 4;
    let mut bboxes: Vec<Vec<Bbox<Vec<KeyPoint>>>> = (0..nclasses).map(|_| vec![]).collect();
    for index in 0..npreds {
        let pred = Vec::<f32>::try_from(pred.i((.., index))?)?;
        let confidence = *pred[4..].iter().max_by(|x, y| x.total_cmp(y)).unwrap();
        if confidence > confidence_threshold {
            let mut class_index = 0;
            for i in 0..nclasses {
                if pred[4 + i] > pred[4 + class_index] {
                    class_index = i
                }
            }
            if pred[class_index + 4] > 0. {
                let bbox = Bbox {
                    xmin: pred[0] - pred[2] / 2.,
                    ymin: pred[1] - pred[3] / 2.,
                    xmax: pred[0] + pred[2] / 2.,
                    ymax: pred[1] + pred[3] / 2.,
                    confidence,
                    data: vec![],
                };
                bboxes[class_index].push(bbox)
            }
        }
    }

    non_maximum_suppression(&mut bboxes, nms_threshold);

    let (initial_h, initial_w) = (
        img.size().unwrap().height as usize,
        img.size().unwrap().width as usize,
    );
    let w_ratio = initial_w as f32 / w as f32;
    let h_ratio = initial_h as f32 / h as f32;
    let mut output_mat = img.clone();
    for (class_index, bboxes_for_class) in bboxes.iter().enumerate() {
        for b in bboxes_for_class.iter() {
            println!("{}: {:?}", classes::NAMES[class_index], b);
            let xmin = (b.xmin * w_ratio) as i32;
            let ymin = (b.ymin * h_ratio) as i32;
            let xmax = (b.xmax * w_ratio) as i32;
            let ymax = (b.ymax * h_ratio) as i32;
            rectangle(
                &mut output_mat,
                Rect::new(xmin, ymin, xmax - xmin, ymax - ymin),
                Scalar::new(0.0, 0.0, 255.0, 0.0), // Red in BGR
                2,
                LINE_8,
                0,
            )
            .unwrap();
            if legend_size > 0 {
                let legend = format!(
                    "{} {:.0}%",
                    classes::NAMES[class_index],
                    100. * b.confidence
                );
                put_text(
                    &mut output_mat,
                    &legend,
                    Point::new(xmin, ymin - 5),
                    FONT_HERSHEY_SIMPLEX,
                    0.5,
                    Scalar::new(255.0, 255.0, 255.0, 0.0), // White
                    1,
                    LINE_AA,
                    false,
                )
                .unwrap();
            }
        }
    }
    Ok(output_mat)
}

pub fn run() -> eyre::Result<()> {
    let device = Device::new_metal(0)?;
    let multiples = Multiples::n();
    let model_file = "model/yolov8n.safetensors";
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[model_file], DType::F32, &device)? };
    let model = YoloV8::load(vb, multiples, 80)?;
    println!("YoloV8n model loaded");

    let mut video_capture = VideoCapture::new(0, CAP_ANY)?;
    if !video_capture.is_opened()? {
        panic!("Unable to open default camera!");
    }

    let window = "Candle - YoloV8n";
    highgui::named_window(window, highgui::WINDOW_AUTOSIZE)?;
    video_capture.set(CAP_PROP_CONVERT_RGB, 1.0)?;

    let target_size = 320;
    let confidence = 0.25;
    let nms = 0.25;
    let legend_size = 14;

    let mut resized_frame = Mat::default();
    let mut rgb_frame = Mat::default();
    loop {
        let mut frame = Mat::default();
        video_capture.read(&mut frame)?;
        if frame.size()?.width > 0 {
            // Resize frame in OpenCV
            let (width, height) = {
                let w = frame.size()?.width as usize;
                let h = frame.size()?.height as usize;
                if w < h {
                    let w = w * target_size / h;
                    (w / 32 * 32, target_size)
                } else {
                    let h = h * target_size / w;
                    (target_size, h / 32 * 32)
                }
            };
            imgproc::resize(
                &frame,
                &mut resized_frame,
                opencv::core::Size::new(width as i32, height as i32),
                0.0,
                0.0,
                imgproc::INTER_NEAREST,
            )?;

            // Convert to RGB for inference
            imgproc::cvt_color(
                &resized_frame,
                &mut rgb_frame,
                imgproc::COLOR_BGR2RGB,
                0,
                opencv::core::AlgorithmHint::ALGO_HINT_DEFAULT,
            )?;

            // Convert to Tensor
            let img_buffer = rgb_frame.data_bytes()?.to_vec();
            let tensor = Tensor::from_vec(img_buffer, (height, width, 3), &device)?
                .to_dtype(DType::F32)?
                .permute((2, 0, 1))?
                .unsqueeze(0)?
                / 255.0;
            let tensor = tensor?;

            // Run inference
            let predictions = model.forward(&tensor)?.squeeze(0)?;
            let processed_mat = YoloV8::report(
                &predictions,
                &frame, // Draw on original frame
                width,
                height,
                confidence,
                nms,
                legend_size,
            )?;

            highgui::imshow(window, &processed_mat)?;
        }

        let key = highgui::wait_key(10)?;
        if key > 0 && key != 255 {
            break;
        }
    }

    video_capture.release()?;
    highgui::destroy_window(window)?;
    Ok(())
}

fn main() -> eyre::Result<()> {
    tracing_subscriber::fmt::init();
    run()?;
    Ok(())
}
