$(document).ready(function(){
    
    const video = document.getElementById("video");

    Promise.all([
        faceapi.nets.ssdMobilenetv1.loadFromUri("models"),
        faceapi.nets.faceLandmark68Net.loadFromUri("models"),
        faceapi.nets.faceRecognitionNet.loadFromUri("models"),
        waitForOpenCV()
    ]).then(startWebcam).then(faceRecognition);

    function waitForOpenCV() {
        return new Promise((resolve) => {
            if (window.cv) {
                // OpenCV already loaded
                cv['onRuntimeInitialized'] = resolve;
            } else {
                // Wait for OpenCV to load
                document.addEventListener('DOMContentLoaded', () => {
                    cv['onRuntimeInitialized'] = resolve;
                });
            }
        });
    }

    function startWebcam()
    {
        navigator.mediaDevices.getUserMedia(
            {
                "video":true, audio:false
            }).then((stream)=>{
                video.srcObject = stream;
            }).catch(()=>{
                console.error(error);
            })
    }

    function getLabeledFaceDescritions()
    {
        const labels = ["Messi", "Ronaldo", "Vijay"];

        return Promise.all(
            labels.map(async (label)=>{

                const descriptions = [];
    
                for (i = 1; i <= 2; i++)
                {
                    const image = await faceapi.fetchImage(`labels/${label}/${i}.png`);
                    
                    const detections = await faceapi.detectSingleFace(image).withFaceLandmarks().withFaceDescriptor();
    
                    descriptions.push(detections.descriptor);
                }
                return new faceapi.LabeledFaceDescriptors(label, descriptions)
            })
        )
    }

    async function checkForSpoofing(videoElement, faceBox) {
        return new Promise((resolve) => {
            const canvas = document.createElement('canvas');
            const ctx = canvas.getContext('2d');
            
            canvas.width = faceBox.width;
            canvas.height = faceBox.height;
            
            ctx.drawImage(
                videoElement,
                faceBox.x, faceBox.y, faceBox.width, faceBox.height,
                0, 0, faceBox.width, faceBox.height
            );
            
            const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
            const src = cv.matFromImageData(imageData);
            const gray = new cv.Mat();
            const laplacian = new cv.Mat();
            const mean = new cv.Mat();
            const stddev = new cv.Mat();
            
            try {
                // Convert to grayscale
                cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY);
                
                // Calculate Laplacian
                cv.Laplacian(gray, laplacian, cv.CV_64F);
                
                // PROPER USAGE OF meanStdDev
                cv.meanStdDev(laplacian, mean, stddev);
                const lapVar = stddev.data64F[0] * stddev.data64F[0]; // Variance = stddev^2
                
                // Calculate gradient magnitude
                const sobelX = new cv.Mat();
                const sobelY = new cv.Mat();
                const gradient = new cv.Mat();
                cv.Sobel(gray, sobelX, cv.CV_64F, 1, 0);
                cv.Sobel(gray, sobelY, cv.CV_64F, 0, 1);
                cv.magnitude(sobelX, sobelY, gradient);
                
                // Get mean gradient magnitude
                const gradMean = cv.mean(gradient).at(0, 0);

                console.log(lapVar+"   "+gradMean);
                
                // Spoof detection thresholds (adjust based on your testing)
                const isSpoofed = lapVar < 85 || gradMean < 20;
                
                resolve(isSpoofed);
            } catch (error) {
                console.error("OpenCV error:", error);
                resolve(false);
            } finally {
                // Clean up all Mats
                src.delete();
                gray.delete();
                laplacian.delete();
                mean.delete();
                stddev.delete();
                if (sobelX) sobelX.delete();
                if (sobelY) sobelY.delete();
                if (gradient) gradient.delete();
            }
        });
    }

    async function faceRecognition()
    {
        const labeledFaceDescriptors = await getLabeledFaceDescritions();
        const faceMatcher = new faceapi.FaceMatcher(labeledFaceDescriptors);

        video.addEventListener('play', async ()=>{

            console.log("playing");
            const canvas = faceapi.createCanvasFromMedia(video);
            document.body.append(canvas);

            const displaySize = {width: video.width, height: video.height};
            faceapi.matchDimensions(canvas, displaySize);

            setInterval(async ()=>{

                const detections = await faceapi.detectAllFaces(video).withFaceLandmarks().withFaceDescriptors();
                const resizedDetections = faceapi.resizeResults(detections, displaySize);

                canvas.getContext('2d').clearRect(0, 0, canvas.width, canvas.height);

                // const results = resizedDetections.map((d)=>{

                //     return faceMatcher.findBestMatch(d.descriptor);
                // });

                // results.forEach((result, i)=>{

                //     const box = resizedDetections[i].detection.box;
                //     const drawBox = new faceapi.draw.DrawBox(box, {label: result});

                //     drawBox.draw(canvas);
                // })
                for (let i = 0; i < resizedDetections.length; i++) {
                    const detection = resizedDetections[i];
                    const box = detection.detection.box;
                    
                    // Perform spoofing check
                    const isSpoofed = await checkForSpoofing(video, box);
                    
                    // Get recognition result
                    const bestMatch = faceMatcher.findBestMatch(detection.descriptor);
                    
                    // Custom label based on spoofing check
                    let label = bestMatch.toString();
                    if (isSpoofed) {
                        label += " (Spoofed!)";
                    }
                    
                    // Draw the box with the combined label
                    const drawBox = new faceapi.draw.DrawBox(box, {
                        label: label,
                        boxColor: isSpoofed ? "red" : "green"
                    });
                    drawBox.draw(canvas);
                }
            }, 100);
        })
    }


// add below code to get blink detection
    // function distance(p1, p2) {// above faceRecognition()
    //     return Math.hypot(p1.x - p2.x, p1.y - p2.y);
    // }
    
    // function getEAR(eye) {// above faceRecognition()
    //     const a = distance(eye[1], eye[5]);
    //     const b = distance(eye[2], eye[4]);
    //     const c = distance(eye[0], eye[3]);
    //     return (a + b) / (2.0 * c);
    // }
// detections.forEach(detection => {// inside setInterval after const detections
//     const leftEye = detection.landmarks.getLeftEye();
//     const rightEye = detection.landmarks.getRightEye();
    
//     const leftEAR = getEAR(leftEye);
//     const rightEAR = getEAR(rightEye);
//     const avgEAR = (leftEAR + rightEAR) / 2;

//     console.log(avgEAR);
    
//     if (avgEAR < 0.23) {
//       console.log("Blink detected!");
//       // You could trigger a UI element or flag this as a "live" face
//     }
// });

});