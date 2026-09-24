import { Router } from 'express';
import { supabase } from '../config/supabase.js';
import { requireAuth, requireRole } from '../middleware/auth.js';

const router = Router();

router.get('/', requireAuth, async (req, res) => {
  const { data, error } = await supabase.from('doctors').select('*, users(full_name, email)');
  if (error) return res.status(500).json({ error: error.message });
  res.json(data);
});

router.get('/:id/patients', requireAuth, requireRole('doctor'), async (req, res) => {
  const { data, error } = await supabase
    .from('patients')
    .select('*, users(full_name, email)')
    .eq('assigned_doctor_id', req.params.id);
  if (error) return res.status(500).json({ error: error.message });
  res.json(data);
});

export default router;
