import { Router } from 'express';
import { supabase } from '../config/supabase.js';
import { requireAuth, requireRole } from '../middleware/auth.js';

const router = Router();

router.get('/', requireAuth, requireRole('doctor'), async (req, res) => {
  const { data, error } = await supabase
    .from('patients')
    .select('*, users(full_name, email), doctors:assigned_doctor_id(id, users(full_name))');
  if (error) return res.status(500).json({ error: error.message });
  res.json(data);
});

router.get('/:id', requireAuth, async (req, res) => {
  const { id } = req.params;
  if (req.user.role === 'patient' && req.user.id !== id) {
    return res.status(403).json({ error: 'Cannot access another patient record' });
  }
  const { data, error } = await supabase
    .from('patients')
    .select('*, users(full_name, email), doctors:assigned_doctor_id(id, users(full_name))')
    .eq('id', id)
    .single();
  if (error) return res.status(404).json({ error: 'Patient not found' });
  res.json(data);
});

router.patch('/:id', requireAuth, requireRole('doctor'), async (req, res) => {
  const { id } = req.params;
  const updates = req.body;
  const { data, error } = await supabase
    .from('patients')
    .update(updates)
    .eq('id', id)
    .select()
    .single();
  if (error) return res.status(500).json({ error: error.message });
  res.json(data);
});

export default router;
