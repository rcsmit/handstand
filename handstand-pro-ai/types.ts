
export interface Point {
  x: number; // 0 to 100 (percentage of width)
  y: number; // 0 to 100 (percentage of height)
}

export interface JointData {
  wrist: Point;
  elbow: Point;
  shoulder: Point;
  hip: Point;
  knee: Point;
  ankle: Point;
}

export interface TorqueData {
  balanceLineX: number; // The vertical line passing through the base (wrists)
  offsets: {
    shoulder: number; // Horizontal distance from balance line
    hip: number;
    ankle: number;
  };
  centerOfMass: Point;
}

export interface HandstandAnalysis {
  leftSide: JointData;
  rightSide: JointData;
  angles: {
    elbow_extension: number;
    shoulder_opening: number;
    hip_alignment: number;
    knee_straightness: number;
  };
  torque: TorqueData;
  score: number;
  feedback: string[];
  classification: 'Perfect' | 'Banana Back' | 'Closed Shoulders' | 'Bent Arms' | 'Piked';
}

export interface AppState {
  image: string | null;
  analyzing: boolean;
  result: HandstandAnalysis | null;
  error: string | null;
}
