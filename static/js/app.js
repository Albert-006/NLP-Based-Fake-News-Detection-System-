document.addEventListener('DOMContentLoaded', () => {
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('image-input');
    const fileName = document.getElementById('file-name');
    const loadingOverlay = document.getElementById('loading-overlay');
    const loadingMessage = document.getElementById('loading-message');
    let activeSubmitButton = null;
    let loaderWatchdog = null;

    const hideLoader = () => {
        if (!loadingOverlay) {
            return;
        }

        loadingOverlay.classList.add('hidden');
        loadingOverlay.classList.remove('flex');
        if (loaderWatchdog) {
            window.clearTimeout(loaderWatchdog);
            loaderWatchdog = null;
        }
        if (activeSubmitButton) {
            activeSubmitButton.disabled = false;
            activeSubmitButton = null;
        }
    };

    const showLoader = (message) => {
        if (!loadingOverlay) {
            return;
        }

        loadingMessage.textContent = message || 'Processing your request...';
        loadingOverlay.classList.remove('hidden');
        loadingOverlay.classList.add('flex');

        if (loaderWatchdog) {
            window.clearTimeout(loaderWatchdog);
        }

        loaderWatchdog = window.setTimeout(() => {
            hideLoader();
            window.alert('The request is taking longer than expected. Please try again.');
        }, 15000);
    };

    window.addEventListener('pageshow', hideLoader);
    window.addEventListener('error', hideLoader);
    window.addEventListener('unhandledrejection', (event) => {
        console.error(event.reason || event);
        hideLoader();
    });

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

    document.querySelectorAll('form').forEach((form) => {
        form.addEventListener('submit', (event) => {
            if (typeof form.reportValidity === 'function' && !form.reportValidity()) {
                hideLoader();
                return;
            }

            const button = event.submitter || form.querySelector('button[type="submit"]');
            if (button) {
                activeSubmitButton = button;
                button.disabled = true;
                showLoader(button.dataset.loadingText);
            } else {
                showLoader();
            }
        });
    });
});
