
import { GoogleGenAI, Type } from "@google/genai";
import { HandstandAnalysis } from "../types";

const ANALYSIS_SCHEMA = {
  type: Type.OBJECT,
  properties: {
    leftSide: {
      type: Type.OBJECT,
      properties: {
        wrist: { type: Type.OBJECT, properties: { x: { type: Type.NUMBER }, y: { type: Type.NUMBER } }, required: ["x", "y"] },
        elbow: { type: Type.OBJECT, properties: { x: { type: Type.NUMBER }, y: { type: Type.NUMBER } }, required: ["x", "y"] },
        shoulder: { type: Type.OBJECT, properties: { x: { type: Type.NUMBER }, y: { type: Type.NUMBER } }, required: ["x", "y"] },
        hip: { type: Type.OBJECT, properties: { x: { type: Type.NUMBER }, y: { type: Type.NUMBER } }, required: ["x", "y"] },
        knee: { type: Type.OBJECT, properties: { x: { type: Type.NUMBER }, y: { type: Type.NUMBER } }, required: ["x", "y"] },
        ankle: { type: Type.OBJECT, properties: { x: { type: Type.NUMBER }, y: { type: Type.NUMBER } }, required: ["x", "y"] },
      },
      required: ["wrist", "elbow", "shoulder", "hip", "knee", "ankle"]
    },
    rightSide: {
      type: Type.OBJECT,
      properties: {
        wrist: { type: Type.OBJECT, properties: { x: { type: Type.NUMBER }, y: { type: Type.NUMBER } }, required: ["x", "y"] },
        elbow: { type: Type.OBJECT, properties: { x: { type: Type.NUMBER }, y: { type: Type.NUMBER } }, required: ["x", "y"] },
        shoulder: { type: Type.OBJECT, properties: { x: { type: Type.NUMBER }, y: { type: Type.NUMBER } }, required: ["x", "y"] },
        hip: { type: Type.OBJECT, properties: { x: { type: Type.NUMBER }, y: { type: Type.NUMBER } }, required: ["x", "y"] },
        knee: { type: Type.OBJECT, properties: { x: { type: Type.NUMBER }, y: { type: Type.NUMBER } }, required: ["x", "y"] },
        ankle: { type: Type.OBJECT, properties: { x: { type: Type.NUMBER }, y: { type: Type.NUMBER } }, required: ["x", "y"] },
      },
      required: ["wrist", "elbow", "shoulder", "hip", "knee", "ankle"]
    },
    angles: {
      type: Type.OBJECT,
      properties: {
        elbow_extension: { type: Type.NUMBER, description: "Angle at elbow in degrees (180 is straight)" },
        shoulder_opening: { type: Type.NUMBER, description: "Angle between torso and arms in degrees" },
        hip_alignment: { type: Type.NUMBER, description: "Angle at hip in degrees" },
        knee_straightness: { type: Type.NUMBER, description: "Angle at knee in degrees" },
      },
      required: ["elbow_extension", "shoulder_opening", "hip_alignment", "knee_straightness"]
    },
    torque: {
      type: Type.OBJECT,
      properties: {
        balanceLineX: { type: Type.NUMBER, description: "The X coordinate (0-100) of the vertical plumb line through the wrists" },
        offsets: {
          type: Type.OBJECT,
          properties: {
            shoulder: { type: Type.NUMBER, description: "Horizontal distance from balance line to shoulders" },
            hip: { type: Type.NUMBER, description: "Horizontal distance from balance line to hips" },
            ankle: { type: Type.NUMBER, description: "Horizontal distance from balance line to ankles" },
          },
          required: ["shoulder", "hip", "ankle"]
        },
        centerOfMass: {
          type: Type.OBJECT,
          properties: {
            x: { type: Type.NUMBER },
            y: { type: Type.NUMBER }
          },
          required: ["x", "y"]
        }
      },
      required: ["balanceLineX", "offsets", "centerOfMass"]
    },
    score: { type: Type.NUMBER, description: "Overall form score from 0-100" },
    feedback: { type: Type.ARRAY, items: { type: Type.STRING } },
    classification: { type: Type.STRING, description: "One of: Perfect, Banana Back, Closed Shoulders, Bent Arms, Piked" },
  },
  required: ["leftSide", "rightSide", "angles", "torque", "score", "feedback", "classification"]
};

export async function analyzeHandstand(base64Image: string): Promise<HandstandAnalysis> {
  // Use safe access for process.env in browser environments
  const apiKey = (window as any).process?.env?.API_KEY || (process as any)?.env?.API_KEY;
  
  if (!apiKey) {
    throw new Error("API Key missing. Please ensure an API key is selected via the Connect button.");
  }

  const ai = new GoogleGenAI({ apiKey });
  const base64Data = base64Image.split(',')[1] || base64Image;

  try {
    const response = await ai.models.generateContent({
      model: "gemini-3-pro-preview", // Complex reasoning multimodal model
      contents: [
        {
          parts: [
            {
              text: `Analyze this handstand photo for professional form and torque. 
              1. Map joints (wrist, elbow, shoulder, hip, knee, ankle) for both sides.
              2. Perform Torque Analysis: Find the vertical 'plumb line' starting from the center of the wrists. 
              3. Calculate horizontal offsets (torque) for the shoulders, hips, and ankles relative to this plumb line.
              4. Estimate the Center of Mass (COM).
              5. Calculate extension angles.
              6. Provide a 'Stack Score' and specific corrective feedback. 
              X and Y are 0-100 percentages.`
            },
            {
              inlineData: {
                mimeType: "image/jpeg",
                data: base64Data,
              }
            }
          ]
        }
      ],
      config: {
        responseMimeType: "application/json",
        responseSchema: ANALYSIS_SCHEMA,
      }
    });

    const resultText = response.text;
    if (!resultText) {
      throw new Error("No analysis received from AI.");
    }

    return JSON.parse(resultText.trim()) as HandstandAnalysis;
  } catch (error: any) {
    console.error("Gemini Analysis Error:", error);
    if (error.message?.includes("Requested entity was not found") || error.message?.includes("API key")) {
        throw new Error("API Key issue. Please re-select your API key.");
    }
    throw new Error(error.message || "The AI analysis failed. Please try a clearer photo.");
  }
}
