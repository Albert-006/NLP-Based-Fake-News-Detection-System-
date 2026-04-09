document.addEventListener('DOMContentLoaded', () => {
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('image-input');
    const fileName = document.getElementById('file-name');
    const loadingOverlay = document.getElementById('loading-overlay');
    const loadingMessage = document.getElementById('loading-message');

    if (dropZone && fileInput) {
        const updateFileName = (file) => {
            fileName.textContent = file ? file.name : '';
        };

        ['dragenter', 'dragover'].forEach((eventName) => {
            dropZone.addEventListener(eventName, (event) => {
                event.preventDefault();
                dropZone.classList.add('dragover');
            });
        });

        ['dragleave', 'drop'].forEach((eventName) => {
            dropZone.addEventListener(eventName, (event) => {
                event.preventDefault();
                dropZone.classList.remove('dragover');
            });
        });

        dropZone.addEventListener('drop', (event) => {
            const files = event.dataTransfer.files;
            if (files && files.length > 0) {
                fileInput.files = files;
                updateFileName(files[0]);
            }
        });

        fileInput.addEventListener('change', () => {
            updateFileName(fileInput.files[0]);
        });
    }

    document.querySelectorAll('form button[type="submit"]').forEach((button) => {
        button.addEventListener('click', () => {
            const form = button.closest('form');
            if (!form) {
                return;
            }

            if (typeof form.reportValidity === 'function' && !form.reportValidity()) {
                return;
            }

            if (loadingOverlay) {
                loadingMessage.textContent = button.dataset.loadingText || 'Processing your request...';
                loadingOverlay.classList.remove('hidden');
                loadingOverlay.classList.add('flex');
            }

            button.disabled = true;
        });
    });
});
