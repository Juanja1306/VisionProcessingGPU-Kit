

export async function applyCanny(inputs) {    
    inputs.forEach(input => {
        // Map input IDs to API parameters
        let paramName = input.id;
        if (paramName === 'kernel') paramName = 'kernel_size';
        if (paramName === 'low') paramName = 'low_threshold';
        if (paramName === 'high') paramName = 'high_threshold';

        // Only append if value > 0 for thresholds (to trigger auto logic)
        if ((paramName.includes('threshold') && input.value > 0) || !paramName.includes('threshold')) {
            formData.append(paramName, input.value);
        }
    });

    try {
        const response = await fetch('/api/canny', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.detail || 'Processing failed');
        }

        const blob = await response.blob();
        const url = URL.createObjectURL(blob);

        processedImg.src = url;
        processedImg.classList.remove('hidden');
        emptyProcessed.classList.add('hidden');

        downloadLink.href = url;
        downloadLink.download = `canny_result_${Date.now()}.png`;
        downloadLink.classList.remove('hidden');

    } catch (error) {
        alert('Error: ' + error.message);
        console.error(error);
    } finally {
        loading.classList.add('hidden');
    }
}