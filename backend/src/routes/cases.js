import { Router } from 'express';
import multer from 'multer';
import { supabase } from '../config/supabase.js';
import { requireAuth, requireRole } from '../middleware/auth.js';

const router = Router();
const upload = multer({ storage: multer.memoryStorage() });

const ML_SERVICE_URL = process.env.ML_SERVICE_URL || 'http://localhost:8000';
const CASE_TYPE_ENDPOINT = {
  vessel_analysis: '/vessel-analysis/predict',
  ef_estimation: '/ef-estimation/predict',
  lv_outline: '/lv-outline/predict',
  dominance: '/dominance/predict',
  arrhythmia: '/arrhythmia/predict',
};

router.get('/', requireAuth, async (req, res) => {
  let query = supabase.from('cases').select('*, patients(id, users(full_name)), case_results(*)');
  if (req.user.role === 'patient') query = query.eq('patient_id', req.user.id);
  const { data, error } = await query.order('created_at', { ascending: false });
  if (error) return res.status(500).json({ error: error.message });
  res.json(data);
});

router.post('/', requireAuth, requireRole('doctor'), upload.single('file'), async (req, res) => {
  const { patient_id, case_type } = req.body;
  if (!patient_id || !case_type) {
    return res.status(400).json({ error: 'patient_id and case_type are required' });
  }

  const { data: newCase, error: caseError } = await supabase
    .from('cases')
    .insert({ patient_id, doctor_id: req.user.id, case_type, status: 'pending' })
    .select()
    .single();
  if (caseError) return res.status(500).json({ error: caseError.message });

  if (req.file) {
    const path = `${newCase.id}/${req.file.originalname}`;
    const { error: uploadError } = await supabase.storage
      .from('case-files')
      .upload(path, req.file.buffer, { contentType: req.file.mimetype });
    if (uploadError) return res.status(500).json({ error: uploadError.message });

    const { data: publicUrl } = supabase.storage.from('case-files').getPublicUrl(path);
    await supabase.from('case_files').insert({
      case_id: newCase.id,
      file_url: publicUrl.publicUrl,
      file_type: req.file.mimetype,
    });
  }

  res.status(201).json(newCase);
});

router.post('/:id/analyze', requireAuth, requireRole('doctor'), async (req, res) => {
  const { id } = req.params;
  const { data: caseRow } = await supabase
    .from('cases')
    .select('*, case_files(*)')
    .eq('id', id)
    .single();
  if (!caseRow) return res.status(404).json({ error: 'Case not found' });

  const endpoint = CASE_TYPE_ENDPOINT[caseRow.case_type];
  await supabase.from('cases').update({ status: 'processing' }).eq('id', id);

  try {
    const mlResponse = await fetch(`${ML_SERVICE_URL}${endpoint}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ file_url: caseRow.case_files[0]?.file_url }),
    });
    const result = await mlResponse.json();

    await supabase.from('case_results').insert({
      case_id: id,
      result_json: result,
      result_summary: result.summary,
      confidence: result.confidence,
    });
    await supabase.from('cases').update({ status: 'completed' }).eq('id', id);
    res.json(result);
  } catch (err) {
    await supabase.from('cases').update({ status: 'failed' }).eq('id', id);
    res.status(502).json({ error: 'ML service unavailable', detail: err.message });
  }
});

export default router;
