// Simple smooth scrolling
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
            target.scrollIntoView({
                behavior: 'smooth',
                block: 'start'
            });
        }
    });
});

// File upload functionality
const API_BASE_URL = 'https://accidentdetection-lx68.onrender.com';
let currentVideoFile = null;

const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('videoFile');
const uploadBtn = document.getElementById('uploadBtn');
const videoContainer = document.getElementById('videoContainer');
const uploadStatus = document.getElementById('uploadStatus');

// Make upload area clickable
uploadArea.addEventListener('click', () => {
    fileInput.click();
});

// File input change
fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        handleFileSelect(e.target.files[0]);
    }
});

// Handle file selection
function handleFileSelect(file) {
    if (!file.type.startsWith('video/')) {
        alert('Please select a valid video file.');
        return;
    }

    currentVideoFile = file;
    uploadBtn.style.display = 'inline-block';
    
    // Show upload status with checkmark
    uploadStatus.style.display = 'block';
}

// Upload button click
uploadBtn.addEventListener('click', uploadVideo);

// Upload video function
async function uploadVideo() {
    if (!currentVideoFile) {
        alert('Please select a video file first.');
        return;
    }

    const email = document.getElementById('emailInput').value.trim();
    if (!email) {
        alert('Please enter your email address.');
        return;
    }

    const formData = new FormData();
    formData.append('file', currentVideoFile);
    formData.append('email', email);

    try {
        uploadBtn.textContent = 'Uploading...';
        uploadBtn.disabled = true;
        uploadStatus.style.display = 'none'; // Hide status during upload

        const response = await fetch(`${API_BASE_URL}/upload_video/`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const result = await response.json();
        
        // Scroll to video container
        videoContainer.scrollIntoView({ 
            behavior: 'smooth', 
            block: 'start' 
        });
        
        // Start the video feed
        setTimeout(() => {
            startVideoFeed();
        }, 1000);

    } catch (err) {
        alert('Backend not connected. Please make sure the server is running.');
    } finally {
        uploadBtn.textContent = 'Upload & Analyze';
        uploadBtn.disabled = false;
    }
}

// Start video feed
function startVideoFeed() {
    // Add timestamp to prevent caching and force fresh request
    const timestamp = new Date().getTime();
    videoContainer.innerHTML = `
        <img src="${API_BASE_URL}/video_feed/?t=${timestamp}" 
             alt="Accident Detection Feed"
             style="width: 100%; height: auto; border-radius: 20px;"
             onerror="alert('Backend not connected. Please make sure the server is running.')">
    `;
}

// LinkedIn integration
function openLinkedIn(profile) {
    const linkedInUrls = {
        'pablo-calderon': 'https://linkedin.com/in/pablo-calderon',
        'pedro-gentil': 'https://linkedin.com/in/pedro-gentil',
        'manuel-ventura': 'https://linkedin.com/in/manuel-ventura',
        'juan-fernandez': 'https://linkedin.com/in/juan-fernandez',
        'adrian-corrochano': 'https://linkedin.com/in/adrian-corrochano'
    };
    
    const url = linkedInUrls[profile];
    if (url) {
        window.open(url, '_blank');
    } else {
        alert('LinkedIn profile not found. Please update the profile URL.');
    }
}