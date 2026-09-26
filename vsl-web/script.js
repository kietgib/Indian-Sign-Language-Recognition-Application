document.addEventListener('DOMContentLoaded', () => {
    const btnStart = document.getElementById('btnStartCamera');
    const btnStop = document.getElementById('btnStopCamera');
    const btnUpload = document.getElementById('btnUploadData');
    const fileInput = document.getElementById('fileInput');
    const statusText = document.querySelector('.status-indicator');
    const videoPlaceholderIcon = document.getElementById('videoPlaceholderIcon');
    const videoPlaceholderText = document.getElementById('videoPlaceholderText');
    const rawVideoPlayer = document.getElementById('rawVideoPlayer');
    const liveResult = document.getElementById('liveResult');
    const uploadStatus = document.getElementById('uploadStatus');

    let animationId;
    let isRunning = false;
    let frameIndex = 0;
    let currentData = [];
    let IS_REAL_DATA = false;
    let currentPrediction = "---";

    // Setup MediaPipe Holistic Client
    const holistic = new Holistic({locateFile: (file) => {
        return `https://cdn.jsdelivr.net/npm/@mediapipe/holistic/${file}`;
    }});
    holistic.setOptions({
        modelComplexity: 1,
        smoothLandmarks: true,
        enableSegmentation: false,
        smoothSegmentation: false,
        refineFaceLandmarks: false,
        minDetectionConfidence: 0.5,
        minTrackingConfidence: 0.5
    });

    // We process local video using a hidden canvas technique
    let localFramesData = [];
    let processingVideo = false;
    const processCanvas = document.createElement('canvas');
    const processCtx = processCanvas.getContext('2d');
    
    // Normalization logic identical to their Python code
    function normalizeKeypoints(pose, face, lh, rh) {
        // Fallback for missing parts
        const makeZeros = (len) => new Array(len).fill(0);
        
        let poseArr = makeZeros(132); // 33 * 4
        let faceArr = makeZeros(1404); // 468 * 3
        let lhArr = makeZeros(63); // 21 * 3
        let rhArr = makeZeros(63); // 21 * 3
        
        if (pose) {
            for(let i=0; i<33; i++) {
                poseArr[i*4] = pose[i].x || 0;
                poseArr[i*4+1] = pose[i].y || 0;
                poseArr[i*4+2] = pose[i].z || 0;
                poseArr[i*4+3] = pose[i].visibility || 0;
            }
        }
        if (face) {
            for(let i=0; i<468; i++) {
                faceArr[i*3] = face[i].x || 0;
                faceArr[i*3+1] = face[i].y || 0;
                faceArr[i*3+2] = face[i].z || 0;
            }
        }
        if (lh) {
            for(let i=0; i<21; i++) {
                lhArr[i*3] = lh[i].x || 0;
                lhArr[i*3+1] = lh[i].y || 0;
                lhArr[i*3+2] = lh[i].z || 0;
            }
        }
        if (rh) {
            for(let i=0; i<21; i++) {
                rhArr[i*3] = rh[i].x || 0;
                rhArr[i*3+1] = rh[i].y || 0;
                rhArr[i*3+2] = rh[i].z || 0;
            }
        }
        
        const lShoulderX = poseArr[44], lShoulderY = poseArr[45];
        const rShoulderX = poseArr[48], rShoulderY = poseArr[49];
        
        if(lShoulderX === 0 && rShoulderX === 0) {
            return [...poseArr, ...faceArr, ...lhArr, ...rhArr];
        }
        
        const centerX = (lShoulderX + rShoulderX) / 2.0;
        const centerY = (lShoulderY + rShoulderY) / 2.0;
        
        const dist = Math.sqrt(Math.pow(rShoulderX - lShoulderX, 2) + Math.pow(rShoulderY - lShoulderY, 2));
        const scale = dist > 1e-6 ? dist : 1.0;
        
        const normXY = (arr, stride) => {
            const out = [...arr];
            for(let i=0; i<out.length; i+=stride) {
                out[i] = (out[i] - centerX) / scale;
                out[i+1] = (out[i+1] - centerY) / scale;
            }
            return out;
        };
        
        return [
            ...normXY(poseArr, 4),
            ...normXY(faceArr, 3),
            ...normXY(lhArr, 3),
            ...normXY(rhArr, 3)
        ];
    }

    holistic.onResults((results) => {
        if (!processingVideo) return;
        const normalized1662 = normalizeKeypoints(
            results.poseLandmarks, 
            results.faceLandmarks, 
            results.leftHandLandmarks, 
            results.rightHandLandmarks
        );
        localFramesData.push(normalized1662);
    });

    if(btnUpload) btnUpload.addEventListener('click', () => fileInput.click());

    if(fileInput) {
        fileInput.addEventListener('change', async (e) => {
            if(!e.target.files || e.target.files.length === 0) return;
            const file = e.target.files[0];
            const isVideo = file.name.toLowerCase().endsWith('.mp4') || file.type.startsWith('video');
            
            statusText.classList.remove('active');
            statusText.innerHTML = '<span class="pulse" style="background:orange"></span> Model: Extracting...';
            stopVis();
            
            if (isVideo) {
                uploadStatus.innerText = "Extracting Keypoints in Browser (0%)...";
                rawVideoPlayer.src = URL.createObjectURL(file);
                rawVideoPlayer.style.display = 'block';
                videoPlaceholderIcon.style.display = 'none';
                videoPlaceholderText.style.display = 'none';
                
                rawVideoPlayer.onloadedmetadata = async () => {
                    processCanvas.width = rawVideoPlayer.videoWidth;
                    processCanvas.height = rawVideoPlayer.videoHeight;
                    localFramesData = [];
                    processingVideo = true;
                    
                    // Simple logic to extract frames every 33ms (~30fps) by seeking video
                    let duration = rawVideoPlayer.duration;
                    let currentTime = 0;
                    
                    rawVideoPlayer.pause();
                    
                    const extractNextFrame = async () => {
                        processCtx.drawImage(rawVideoPlayer, 0, 0, processCanvas.width, processCanvas.height);
                        await holistic.send({image: processCanvas});
                        
                        let pct = Math.floor((currentTime / duration) * 100);
                        uploadStatus.innerText = `Extracting Keypoints... ${pct}%`;
                        
                        currentTime += 0.0333; // 30 FPS
                        if(currentTime <= duration) {
                            rawVideoPlayer.currentTime = currentTime;
                        } else {
                            // Extraction done, Send to Backend
                            processingVideo = false;
                            uploadStatus.innerText = "Processing Sequence in Python...";
                            
                            try {
                                const formData = new FormData();
                                const blob = new Blob([new Float32Array(localFramesData.flat()).buffer], { type: 'application/octet-stream' });
                                formData.append('file', blob, 'sequence.npy');
                                
                                const req = await fetch('http://localhost:5000/predict_npy', { method: 'POST', body: formData });
                                const result = await req.json();
                                
                                if(result.success) {
                                    currentData = localFramesData;
                                    IS_REAL_DATA = true;
                                    currentPrediction = result.predictions[0].label;
                                    
                                    const bars = document.querySelectorAll('.pred-item');
                                    for(let i=0; i<3; i++) {
                                        if(result.predictions[i] && bars[i]) {
                                            bars[i].querySelector('.pred-label').innerHTML = `<span>${result.predictions[i].label}</span> <span>${result.predictions[i].confidence.toFixed(1)}%</span>`;
                                            bars[i].querySelector('.progress-fill').style.width = `${result.predictions[i].confidence}%`;
                                        }
                                    }
                                    
                                    uploadStatus.innerText = "Analysis Complete!";
                                    rawVideoPlayer.currentTime = 0;
                                    rawVideoPlayer.play();
                                    startVis();
                                } else {
                                    uploadStatus.innerText = result.error;
                                }
                            } catch(ex) {
                                uploadStatus.innerText = "Failed to contact Python backend.";
                            }
                        }
                    };
                    
                    rawVideoPlayer.onseeked = () => {
                        if(processingVideo) extractNextFrame();
                    };
                    
                    rawVideoPlayer.currentTime = 0; // Starts the extraction loop
                };
            }
        });
    }

    function startVis() {
        if(isRunning || !IS_REAL_DATA) return;
        isRunning = true;
        frameIndex = 0;
        
        statusText.classList.add('active');
        statusText.innerHTML = '<span class="pulse"></span> Model: Active (Real Inference)';
        
        const existingCanvas = document.getElementById('overlayCanvas');
        if (!existingCanvas) {
            const canvasHTML = '<canvas id="overlayCanvas"></canvas>';
            document.getElementById('skeletonFeed').insertAdjacentHTML('afterbegin', canvasHTML);
        }
        
        const newCanvas = document.getElementById('overlayCanvas');
        startSimulatingMediaPipe(newCanvas, newCanvas.getContext('2d'));
    }

    function stopVis() {
        isRunning = false;
        if(animationId) cancelAnimationFrame(animationId);
        
        statusText.classList.remove('active');
        statusText.innerHTML = '<span class="pulse"></span> Model: Standby';
        
        if (rawVideoPlayer) rawVideoPlayer.pause();
        
        liveResult.innerText = "---";
        liveResult.style.color = "var(--text-muted)";
    }

    if (btnStart) btnStart.addEventListener('click', () => { 
        if(rawVideoPlayer && rawVideoPlayer.src) { rawVideoPlayer.currentTime = 0; rawVideoPlayer.play(); }
        startVis(); 
    });
    if (btnStop) btnStop.addEventListener('click', stopVis);

    function startSimulatingMediaPipe(canvas, context) {
        const setSize = () => {
            if(!canvas.parentElement) return;
            canvas.width = canvas.parentElement.clientWidth;
            canvas.height = canvas.parentElement.clientHeight;
        };
        setSize();
        window.addEventListener('resize', setSize);
        
        function draw() {
            if(!isRunning) return;
            context.clearRect(0, 0, canvas.width, canvas.height);
            
            const cx = canvas.width / 2;
            const cy = canvas.height * 0.4;
            
            // Drop scale significantly so the whole body fits inside the tracking view
            const scale = Math.min(canvas.width, canvas.height) * 0.18; 
            
            const tPose = document.getElementById('togglePose')?.checked ?? true;
            const tHands = document.getElementById('toggleHands')?.checked ?? true;
            const tFace = document.getElementById('toggleFace')?.checked ?? true;
            
            if(rawVideoPlayer && rawVideoPlayer.style.display === 'block') {
                const fps = 30; 
                frameIndex = Math.min(Math.floor(rawVideoPlayer.currentTime * fps), currentData.length - 1);
            }
            
            if(currentData.length === 0) return;
            const frame = currentData[frameIndex];
            
            const rX = (val) => cx + (val * scale);
            const rY = (val) => cy + (val * scale);
            
            const drawLine = (idxA, idxB, offset, stride, color, width=2) => {
               const a = offset + (idxA * stride);
               const b = offset + (idxB * stride);
               if (frame[a] === 0 && frame[a+1] === 0) return;
               context.beginPath();
               context.strokeStyle = color; context.lineWidth = width;
               context.moveTo(rX(frame[a]), rY(frame[a+1]));
               context.lineTo(rX(frame[b]), rY(frame[b+1]));
               context.stroke();
            };
            
            const drawPoints = (start, count, stride, color, rad=3) => {
                context.fillStyle = color;
                for(let i=0; i<count; i++) {
                    const px = rX(frame[start + i*stride]);
                    const py = rY(frame[start + i*stride + 1]);
                    if(px === cx && py === cy && frame[start+i*stride]==0) continue;
                    context.beginPath();
                    context.arc(px, py, rad, 0, 2*Math.PI);
                    context.fill();
                }
            }

            if(tPose) {
                drawLine(11, 12, 0, 4, 'rgba(255,255,255,0.7)', 4);
                drawLine(11, 13, 0, 4, 'rgba(255,255,255,0.7)', 4);
                drawLine(13, 15, 0, 4, 'rgba(255,255,255,0.7)', 4);
                drawLine(12, 14, 0, 4, 'rgba(255,255,255,0.7)', 4);
                drawLine(14, 16, 0, 4, 'rgba(255,255,255,0.7)', 4);
                drawPoints(0, 33, 4, '#fff', 4);
            }
            if(tHands) {
                drawPoints(1536, 21, 3, '#fca5a5', 3);
                drawPoints(1599, 21, 3, '#4ade80', 3);
            }
            if(tFace) drawPoints(132, 468, 3, '#3b82f6', 1);
            
            const textThresh = Math.floor(currentData.length * 0.2);
            if (frameIndex > textThresh) {
                if(liveResult) liveResult.innerText = currentPrediction;
                if(liveResult) { liveResult.style.color = "var(--primary)"; liveResult.style.textShadow = "0 0 15px var(--primary-glow)"; }
            } else {
                if(liveResult) liveResult.innerText = "Analyzing...";
                if(liveResult) { liveResult.style.color = "var(--text-muted)"; liveResult.style.textShadow = "none"; }
            }
            
            animationId = requestAnimationFrame(draw);
        }
        draw();
    }
    
    // Quick backend health check & Classes loading
    fetch('http://localhost:5000/ping').then(r=>r.json()).then(d=>{
        if(uploadStatus) uploadStatus.innerHTML = "<span style='color:#4ade80'>Backend Online (Ready)</span>";
    }).catch(e=>{
        if(uploadStatus) uploadStatus.innerHTML = "<span style='color:#ef4444'>Backend Offline. Mở Terminal chạy: python server.py</span>";
    });
});
