/* ═══════════════════════════════════════════════
   AuthKYC — Frontend Application Logic
   Handles upload, API call, and results rendering
   ═══════════════════════════════════════════════ */

const API_URL = '/api/v1/audit_stream';

// ── DOM Elements ──
const uploadZone    = document.getElementById('uploadZone');
const fileInput     = document.getElementById('fileInput');
const browseBtn     = document.getElementById('browseBtn');
const fileInfo      = document.getElementById('fileInfo');
const fileName      = document.getElementById('fileName');
const fileSize      = document.getElementById('fileSize');
const analyzeBtn    = document.getElementById('analyzeBtn');
const uploadSection = document.getElementById('uploadSection');
const processingSection = document.getElementById('processingSection');
const resultsSection    = document.getElementById('resultsSection');
const progressFill  = document.getElementById('progressFill');
const processingStatus = document.getElementById('processingStatus');
const resetBtn      = document.getElementById('resetBtn');

let selectedFile = null;

// ── Status Messages for Processing Animation ──
const statusMessages = [
  'Initializing PRNU sensor forensics...',
  'Accumulating camera noise residuals...',
  'Computing 2D FFT for Moiré detection...',
  'Extracting rPPG chrominance signals...',
  'Buffering biological pulse waveform...',
  'Running MTCNN face detection...',
  'Assembling 16-frame tensor sequence...',
  'FTCA cross-attention inference...',
  'Applying waterfall decision logic...'
];

// ═══════════════ FILE SELECTION ═══════════════

uploadZone.addEventListener('click', () => fileInput.click());
browseBtn.addEventListener('click', (e) => { e.stopPropagation(); fileInput.click(); });

uploadZone.addEventListener('dragover', (e) => {
  e.preventDefault();
  uploadZone.classList.add('dragover');
});

uploadZone.addEventListener('dragleave', () => {
  uploadZone.classList.remove('dragover');
});

uploadZone.addEventListener('drop', (e) => {
  e.preventDefault();
  uploadZone.classList.remove('dragover');
  const file = e.dataTransfer.files[0];
  if (file) handleFileSelection(file);
});

fileInput.addEventListener('change', () => {
  if (fileInput.files[0]) handleFileSelection(fileInput.files[0]);
});

function handleFileSelection(file) {
  const validExts = ['.mp4', '.avi', '.mov', '.webm'];
  const ext = '.' + file.name.split('.').pop().toLowerCase();
  if (!validExts.includes(ext)) {
    alert('Unsupported format. Please use .mp4, .avi, .mov, or .webm');
    return;
  }
  selectedFile = file;
  fileName.textContent = file.name;
  fileSize.textContent = formatBytes(file.size);
  fileInfo.style.display = 'flex';
}

function formatBytes(bytes) {
  if (bytes < 1024) return bytes + ' B';
  if (bytes < 1048576) return (bytes / 1024).toFixed(1) + ' KB';
  return (bytes / 1048576).toFixed(1) + ' MB';
}

// ═══════════════ ANALYSIS ═══════════════

analyzeBtn.addEventListener('click', () => {
  if (!selectedFile) return;
  startAnalysis();
});

async function startAnalysis() {
  // Show processing
  uploadSection.style.display = 'none';
  resultsSection.style.display = 'none';
  processingSection.style.display = 'block';

  // Animate status messages
  let msgIndex = 0;
  const statusInterval = setInterval(() => {
    msgIndex = (msgIndex + 1) % statusMessages.length;
    processingStatus.textContent = statusMessages[msgIndex];
  }, 1500);

  // Animate progress bar (fake progress until response)
  let progress = 0;
  const progressInterval = setInterval(() => {
    progress = Math.min(progress + Math.random() * 8, 90);
    progressFill.style.width = progress + '%';
  }, 500);

  try {
    const formData = new FormData();
    formData.append('video', selectedFile);

    const response = await fetch(API_URL, {
      method: 'POST',
      body: formData
    });

    clearInterval(statusInterval);
    clearInterval(progressInterval);
    progressFill.style.width = '100%';

    if (!response.ok) {
      const err = await response.json();
      throw new Error(err.detail || `Server error: ${response.status}`);
    }

    const data = await response.json();

    setTimeout(() => {
      processingSection.style.display = 'none';
      renderResults(data);
    }, 400);

  } catch (err) {
    clearInterval(statusInterval);
    clearInterval(progressInterval);
    processingSection.style.display = 'none';
    uploadSection.style.display = 'block';
    alert('Analysis failed: ' + err.message);
  }
}

// ═══════════════ RESULTS RENDERING ═══════════════

function renderResults(data) {
  resultsSection.style.display = 'block';

  // Verdict
  const verdictCard = document.getElementById('verdictCard');
  const isApproved = data.final_decision === 'APPROVED';
  verdictCard.className = 'verdict-card ' + (isApproved ? 'approved' : 'denied');

  document.getElementById('verdictIcon').textContent = isApproved ? '✅' : '🛑';
  document.getElementById('verdictTitle').textContent = isApproved ? 'SESSION APPROVED' : data.final_decision;
  document.getElementById('verdictDetail').textContent = isApproved
    ? 'All 4 security stages passed — identity verified'
    : 'Session terminated by the security waterfall';
  document.getElementById('verdictTime').textContent = data.processing_time_seconds + 's';

  // Telemetry
  document.getElementById('telemetryText').textContent = 'Stream Telemetry: ' + data.telemetry_status;

  // S1
  renderStage('stage1Card', 's1', data.stage_1_sensor_prnu, {
    scoreFmt: v => v.toFixed(2),
    barPct: Math.min(data.stage_1_sensor_prnu.score / 5, 100)
  });

  // S2
  renderStage('stage2Card', 's2', data.stage_2_presentation_replay, {
    scoreFmt: v => v.toFixed(0),
    barPct: Math.min(data.stage_2_presentation_replay.score / 120, 100)
  });

  // S3
  renderStage('stage3Card', 's3', data.stage_3_biological_rppg, {
    scoreFmt: v => v.toFixed(2) + ' dB',
    barPct: Math.min(data.stage_3_biological_rppg.score / 10, 100)
  });

  // Extract BPM from details string
  const bpmMatch = data.stage_3_biological_rppg.details.match(/Pulse: ([\d.]+) BPM/);
  document.getElementById('s3Bpm').textContent = bpmMatch ? bpmMatch[1] + ' BPM' : '—';

  // S4
  renderStage('stage4Card', 's4', data.stage_4_synthetic_ftca, {
    scoreFmt: v => v.toFixed(4),
    barPct: data.stage_4_synthetic_ftca.score * 100
  });

  // Raw JSON
  document.getElementById('rawJson').textContent = JSON.stringify(data, null, 2);
}

function renderStage(cardId, prefix, stageData, opts) {
  const card = document.getElementById(cardId);
  const passed = stageData.passed;

  card.classList.remove('pass', 'fail');
  card.classList.add(passed ? 'pass' : 'fail');

  const badge = document.getElementById(prefix + 'Badge');
  badge.textContent = passed ? 'PASS' : 'FAIL';
  badge.className = 'stage-badge ' + (passed ? 'pass' : 'fail');

  document.getElementById(prefix + 'Score').textContent = opts.scoreFmt(stageData.score);
  document.getElementById(prefix + 'Detail').textContent = stageData.details;

  // Animate bar
  const bar = document.getElementById(prefix + 'Bar');
  setTimeout(() => {
    bar.style.width = Math.max(opts.barPct, 3) + '%';
  }, 100);
}

// ═══════════════ RESET ═══════════════

resetBtn.addEventListener('click', () => {
  resultsSection.style.display = 'none';
  uploadSection.style.display = 'block';
  fileInfo.style.display = 'none';
  fileInput.value = '';
  selectedFile = null;
  progressFill.style.width = '0%';

  // Reset bars
  ['s1Bar', 's2Bar', 's3Bar', 's4Bar'].forEach(id => {
    document.getElementById(id).style.width = '0%';
  });
});
