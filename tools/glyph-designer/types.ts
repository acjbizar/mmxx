
export type Position = 'top' | 'right' | 'bottom' | 'left';

export interface PolygonData {
  id: string;
  points: string;
  row: number;
  col: number;
  pos: Position;
}

export interface AppState {
  activeIds: Set<string>;
}
