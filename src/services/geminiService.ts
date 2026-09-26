import { GoogleGenAI, Type } from '@google/genai';
import { CropDiagnosis } from '../types';

/**
 * Real AI Crop Doctor — Gemini Vision integration.
 *
 * Replaces the static SAMPLE_DIAGNOSES preset flow with an actual
 * multimodal call to Gemini: the farmer's leaf photo is sent to the
 * model along with a structured schema, and Gemini returns a real
 * diagnosis (disease, confidence, treatment plan, bilingual narration)
 * in the exact shape the rest of the app already expects.
 *
 * SECURITY NOTE: This key is read from a Vite `VITE_`-prefixed env var,
 * which means it ships inside the client bundle and is visible to
 * anyone who inspects the app. That's fine for a prototype / hackathon
 * build, but before a real production launch this call should move
 * behind a small backend proxy (e.g. a Cloud Run / Express endpoint)
 * that holds the key server-side and forwards only the image.
 */

const API_KEY = import.meta.env.VITE_GEMINI_API_KEY as string | undefined;

let client: GoogleGenAI | null = null;

/** Whether a Gemini API key has been configured for this build. */
export const isGeminiConfigured = (): boolean => Boolean(API_KEY && API_KEY.trim().length > 0);

const getClient = (): GoogleGenAI => {
  if (!client) {
    if (!API_KEY) {
      throw new Error('GEMINI_NOT_CONFIGURED');
    }
    client = new GoogleGenAI({ apiKey: API_KEY });
  }
  return client;
};

// Structured output schema — mirrors the CropDiagnosis type in src/types.ts
// so Gemini's JSON response can be used directly, with no manual mapping.
const diagnosisSchema = {
  type: Type.OBJECT,
  properties: {
    cropName: { type: Type.STRING, description: 'Common crop name, e.g. "Tomato"' },
    diseaseName: { type: Type.STRING, description: 'Common disease/pest name, e.g. "Early Blight"' },
    scientificName: { type: Type.STRING, description: 'Pathogen scientific name, e.g. "Alternaria solani". Use "N/A" if healthy.' },
    severity: { type: Type.STRING, enum: ['high', 'medium', 'low'] },
    confidence: { type: Type.NUMBER, description: 'Diagnostic confidence as a percentage, 0-100' },
    detectedVisual: { type: Type.STRING, description: 'A single emoji representing the crop or symptom' },
    urgencyDays: { type: Type.STRING, description: 'How soon treatment is needed, e.g. "Within 2-3 days"' },
    symptoms: { type: Type.ARRAY, items: { type: Type.STRING }, description: '3-5 visible symptoms detected in the photo' },
    treatmentSteps: {
      type: Type.ARRAY,
      items: { type: Type.STRING },
      description: '3-5 concrete treatment steps, mixing chemical dosage (e.g. "Mancozeb 75% WP @ 2.5g/L") and organic alternatives (e.g. "Neem oil @ 5ml/L")'
    },
    recommendedProduct: {
      type: Type.OBJECT,
      properties: {
        name: { type: Type.STRING },
        dosage: { type: Type.STRING },
        price: { type: Type.NUMBER, description: 'Approximate price in INR' }
      },
      required: ['name', 'dosage', 'price']
    },
    hindiNarration: { type: Type.STRING, description: 'A short 2-3 sentence spoken summary of the diagnosis and remedy, in Hindi (Devanagari script)' },
    englishNarration: { type: Type.STRING, description: 'The same 2-3 sentence summary in simple Indian English' }
  },
  required: [
    'cropName',
    'diseaseName',
    'scientificName',
    'severity',
    'confidence',
    'detectedVisual',
    'urgencyDays',
    'symptoms',
    'treatmentSteps',
    'recommendedProduct',
    'hindiNarration',
    'englishNarration'
  ]
};

const SYSTEM_PROMPT = `You are an expert plant pathologist and agronomist advising smallholder Indian farmers.
Examine the leaf/crop photo provided and identify any disease, pest damage, or nutrient deficiency.
If the plant looks healthy, say so clearly (diseaseName: "Healthy", severity: "low").
Keep advice practical and affordable for a small farm in India: prefer widely available chemical
treatments (with correct dosage) and common organic alternatives (neem oil, Trichoderma, etc).
Respond ONLY with JSON matching the provided schema — no extra commentary.`;

/**
 * Strip the data-URL prefix ("data:image/jpeg;base64,") if present,
 * returning just the base64 payload Gemini expects.
 */
const toBase64Payload = (dataUrlOrBase64: string): string =>
  dataUrlOrBase64.includes(',') ? dataUrlOrBase64.split(',')[1] : dataUrlOrBase64;

/**
 * Send a leaf photo to Gemini Vision and get back a structured CropDiagnosis.
 * Throws if no API key is configured or the request/parse fails —
 * callers should catch this and fall back to a demo preset.
 */
export async function diagnoseLeafImage(
  imageDataUrlOrBase64: string,
  mimeType: string = 'image/jpeg'
): Promise<CropDiagnosis> {
  const ai = getClient();
  const imageData = toBase64Payload(imageDataUrlOrBase64);

  const response = await ai.models.generateContent({
    model: 'gemini-2.5-flash',
    contents: [
      {
        role: 'user',
        parts: [
          { inlineData: { mimeType, data: imageData } },
          { text: SYSTEM_PROMPT }
        ]
      }
    ],
    config: {
      responseMimeType: 'application/json',
      responseSchema: diagnosisSchema
    }
  });

  const rawText = response.text;
  if (!rawText) {
    throw new Error('EMPTY_GEMINI_RESPONSE');
  }

  const parsed = JSON.parse(rawText);

  const diagnosis: CropDiagnosis = {
    id: `diag-${Date.now()}`,
    cropName: parsed.cropName,
    diseaseName: parsed.diseaseName,
    scientificName: parsed.scientificName,
    severity: parsed.severity,
    confidence: Math.round(Number(parsed.confidence)),
    detectedVisual: parsed.detectedVisual,
    urgencyDays: parsed.urgencyDays,
    symptoms: parsed.symptoms,
    treatmentSteps: parsed.treatmentSteps,
    recommendedProduct: parsed.recommendedProduct,
    hindiNarration: parsed.hindiNarration,
    englishNarration: parsed.englishNarration
  };

  return diagnosis;
}
