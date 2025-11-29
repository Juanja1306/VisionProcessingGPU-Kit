import { applyCanny } from './applyCanny.js'
import { applyEmboss } from './applyEmboss.js'

//TODO: estandarizar los nombres

export const filtersNames = {
    CANNY: 'canny',
    GAUSSIAN: 'gaussian',
    SOBEL: 'sobel', // Mapea al data-filter="sobel"
    EMBOSS: 'laplacian', // Mapea al data-filter="laplacian"

};

const applyFilter = {
    [filtersNames.CANNY]: async (formData, inputs) => applyCanny(formData, inputs),
    [filtersNames.GAUSSIAN]: async () => { /* ... */ },
    [filtersNames.NEGATIVE]: async () => { /* ... */ },
    [filtersNames.EMBOSS]: async (formData, inputs) => applyEmboss(formData, inputs),
}


const filterConfig = {
    [filtersNames.CANNY]: {
        name: "Canny Edge Detection",
        controls: [
            { id: 'kernel', label: 'Kernel Size', type: 'range', min: 3, max: 15, step: 2, value: 5 },
            { id: 'sigma', label: 'Sigma', type: 'range', min: 0.1, max: 5.0, step: 0.1, value: 1.4 },
            { id: 'low', label: 'Low Threshold', type: 'range', min: 0, max: 255, step: 1, value: 0, suffix: '(Auto if 0)' },
            { id: 'high', label: 'High Threshold', type: 'range', min: 0, max: 255, step: 1, value: 0, suffix: '(Auto if 0)' }
        ]
    },
    [filtersNames.GAUSSIAN]: {
        name: "Gaussian Blur",
        controls: [
            { id: 'kernel', label: 'Kernel Size', type: 'range', min: 3, max: 15, step: 2, value: 5 },
            { id: 'sigma', label: 'Sigma', type: 'range', min: 0.1, max: 10.0, step: 0.1, value: 2.0 }
        ]
    },
    [filtersNames.SOBEL]: {
        name: "Negative",
        controls: [
            { id: 'ksize', label: 'Kernel Size', type: 'range', min: 1, max: 7, step: 2, value: 3 },
            { id: 'scale', label: 'Scale', type: 'range', min: 1, max: 5, step: 1, value: 1 }
        ]
    },
    [filtersNames.EMBOSS]: {
        name: "Emboss",
        controls: [
            { id: 'ksize', label: 'Kernel Size', type: 'range', min: 1, max: 7, step: 2, value: 3 }
        ]
    }
};

// State
let currentFilter = filtersNames.CANNY;
let currentFile = null;
const controlValues = {}; // Store values to persist when switching

// DOM Elements
const controlsContainer = document.getElementById('controls-container');
const filterTitle = document.getElementById('filter-title');
const navItems = document.querySelectorAll('.nav-item');
const dropArea = document.getElementById('drop-area');
const fileInput = document.getElementById('file-input');
const originalImg = document.getElementById('original-image');
const processedImg = document.getElementById('processed-image');
const emptyOriginal = document.getElementById('empty-original');
const emptyProcessed = document.getElementById('empty-processed');
const loading = document.getElementById('loading');
const processBtn = document.getElementById('process-btn');
const downloadLink = document.getElementById('download-link');


// Initialize
function init() {
    renderControls(currentFilter);
    setupEventListeners();
}

function renderControls(filterKey) {
    const config = filterConfig[filterKey];
    filterTitle.textContent = config.name;
    controlsContainer.innerHTML = '';

    config.controls.forEach(ctrl => {
        // Use stored value if exists, else default
        const val = controlValues[ctrl.id] !== undefined ? controlValues[ctrl.id] : ctrl.value;

        const group = document.createElement('div');
        group.className = 'control-group';
        group.innerHTML = `
                    <div class="control-header">
                        <label>${ctrl.label} ${ctrl.suffix || ''}</label>
                        <span id="val-${ctrl.id}">${val}</span>
                    </div>
                    <input type="${ctrl.type}" id="${ctrl.id}" 
                           min="${ctrl.min}" max="${ctrl.max}" step="${ctrl.step}" value="${val}">
                `;
        controlsContainer.appendChild(group);

        // Event listener for value update
        const input = group.querySelector('input');
        input.addEventListener('input', (e) => {
            document.getElementById(`val-${ctrl.id}`).textContent = e.target.value;
            controlValues[ctrl.id] = e.target.value;
        });
    });
}

function setupEventListeners() {
    // Navigation
    navItems.forEach(item => {
        item.addEventListener('click', () => {
            // Update UI
            navItems.forEach(n => n.classList.remove('active'));
            item.classList.add('active');

            // Update State
            currentFilter = item.dataset.filter;
            renderControls(currentFilter);
        });
    });

    // File Upload
    dropArea.addEventListener('click', () => fileInput.click());

    dropArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropArea.classList.add('drag-over');
    });

    dropArea.addEventListener('dragleave', () => dropArea.classList.remove('drag-over'));

    dropArea.addEventListener('drop', (e) => {
        e.preventDefault();
        dropArea.classList.remove('drag-over');
        if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length) handleFile(e.target.files[0]);
    });

    // Process Button
    processBtn.addEventListener('click', processImage);
}

function handleFile(file) {
    if (!file.type.startsWith('image/')) {
        alert('Please upload an image file.');
        return;
    }
    currentFile = file;

    const reader = new FileReader();
    reader.onload = (e) => {
        originalImg.src = e.target.result;
        originalImg.classList.remove('hidden');
        emptyOriginal.classList.add('hidden');

        // Reset processed
        processedImg.classList.add('hidden');
        emptyProcessed.classList.remove('hidden');
        downloadLink.classList.add('hidden');
    };
    reader.readAsDataURL(file);
}

async function processImage() {
    if (!currentFile) {
        alert('Please upload an image first.');
        return;
    }

    loading.classList.remove('hidden');

    const formData = new FormData();
    formData.append('file', currentFile);
    // Get current values from inputs
    const inputs = controlsContainer.querySelectorAll('input');


    try {
        // 🚨 CAMBIO CLAVE: Pasar BOTH (formData y inputs)
        await applyFilter[currentFilter](formData, inputs);

    } catch (error) {
        alert('Error: ' + error.message);
        console.error(error);
    } finally {
        loading.classList.add('hidden');
    }

}

// Start
init();