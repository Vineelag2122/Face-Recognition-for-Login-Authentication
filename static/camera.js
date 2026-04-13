async function startCamera(videoElement, statusElement) {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user' }, audio: false });
        videoElement.srcObject = stream;
        await videoElement.play();
        if (statusElement) {
            statusElement.textContent = 'Camera is active. Center your face in the frame.';
        }
    } catch (error) {
        if (statusElement) {
            statusElement.textContent = 'Camera access failed. Please allow camera permission.';
        }
        throw error;
    }
}

function frameToDataUrl(videoElement, canvasElement) {
    const context = canvasElement.getContext('2d');
    canvasElement.width = videoElement.videoWidth;
    canvasElement.height = videoElement.videoHeight;
    context.drawImage(videoElement, 0, 0, canvasElement.width, canvasElement.height);
    return canvasElement.toDataURL('image/jpeg', 0.95);
}

function captureFrame(videoElement, canvasElement, hiddenInput, previewElement, statusElement) {
    const dataUrl = frameToDataUrl(videoElement, canvasElement);
    hiddenInput.value = dataUrl;
    if (previewElement) {
        previewElement.src = dataUrl;
        previewElement.classList.remove('d-none');
    }
    if (statusElement) {
        statusElement.textContent = 'Frame captured. You can submit now.';
    }
}

function bindAutoCaptureOnSubmit(formElement, videoElement, canvasElement, hiddenInput, statusElement) {
    formElement.addEventListener('submit', (event) => {
        if (!videoElement.videoWidth || !videoElement.videoHeight) {
            event.preventDefault();
            if (statusElement) {
                statusElement.textContent = 'Camera is not ready yet. Please wait a second and try again.';
            }
            return;
        }

        hiddenInput.value = frameToDataUrl(videoElement, canvasElement);
        if (statusElement) {
            statusElement.textContent = 'Analyzing live camera frame...';
        }
    });
}

window.FaceCamera = {
    startCamera,
    captureFrame,
    bindAutoCaptureOnSubmit,
};
