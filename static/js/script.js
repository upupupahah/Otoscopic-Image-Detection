document.addEventListener('DOMContentLoaded', () => {
    // Проверка существования элементов
    const video = document.getElementById('video');
    if (!video) {
        console.error('ERROR: Video element not found! Check HTML markup');
        return;
    }

    const preview = document.getElementById('preview');
    const btnCamera = document.getElementById('btnCamera');
    const btnFile = document.getElementById('btnFile');
    const btnUpload = document.getElementById('btnUpload');
    const fileInput = document.getElementById('fileInput');
    const response = document.getElementById('response');
    const placeholder = document.querySelector('.placeholder');

    let mediaStream = null;

    // Инициализация
    function init() {
        btnCamera.addEventListener('click', toggleCamera);
        btnFile.addEventListener('click', () => fileInput.click());
        fileInput.addEventListener('change', handleFile);
        btnUpload.addEventListener('click', uploadImage);
    }

    // Работа с камерой
    async function toggleCamera() {
        try {
            if (mediaStream) {
                stopCamera();
                btnCamera.innerHTML = '<i class="fas fa-camera"></i> Enable Camera';
            } else {
                mediaStream = await navigator.mediaDevices.getUserMedia({
                    video: { facingMode: 'environment' }
                });
                video.srcObject = mediaStream;
                video.hidden = false;
                preview.hidden = true;
                placeholder.hidden = true;
                btnCamera.innerHTML = '<i class="fas fa-stop"></i> Stop Camera';
            }
        } catch (err) {
            alert(`Camera error: ${err.message}`);
            console.error('Camera error:', err);
        }
    }

    function stopCamera() {
        if (mediaStream) {
            mediaStream.getTracks().forEach(track => track.stop());
            mediaStream = null;
        }
        video.hidden = true;
        placeholder.hidden = false;
    }

    // Обработка файла
    function handleFile(e) {
        const file = e.target.files[0];
        if (!file || !file.type.startsWith('image/')) return;

        preview.src = URL.createObjectURL(file);
        preview.hidden = false;
        video.hidden = true;
        placeholder.hidden = true;
        btnUpload.disabled = false;
    }

    // Отправка на сервер
    async function uploadImage() {
        try {
            const formData = new FormData();
            const blob = await fetch(preview.src).then(r => r.blob());

            formData.append('file', blob, 'image.jpg');

            const res = await fetch('/predict', {
                method: 'POST',
                body: formData
            });

            const data = await res.json();
            response.textContent = JSON.stringify(data, null, 2);

        } catch (err) {
            response.textContent = `Error: ${err.message}`;
            console.error('Upload error:', err);
        }
    }

    init();
});