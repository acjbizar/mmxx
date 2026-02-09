
import { PolygonData, Position } from './types';

const generatePolygons = (): PolygonData[] => {
  const polys: PolygonData[] = [];
  const positions: Position[] = ['top', 'right', 'bottom', 'left'];

  for (let r = 0; r < 8; r++) {
    for (let c = 0; c < 8; c++) {
      const x = c * 30;
      const y = r * 30;
      const midX = x + 15;
      const midY = y + 15;

      positions.forEach((pos) => {
        let points = '';
        if (pos === 'top') points = `${x},${y} ${x + 30},${y} ${midX},${midY}`;
        else if (pos === 'right') points = `${x + 30},${y} ${x + 30},${y + 30} ${midX},${midY}`;
        else if (pos === 'bottom') points = `${x + 30},${y + 30} ${x},${y + 30} ${midX},${midY}`;
        else if (pos === 'left') points = `${x},${y + 30} ${x},${y} ${midX},${midY}`;

        polys.push({
          id: `r${r}c${c}-${pos}`,
          points,
          row: r,
          col: c,
          pos
        });
      });
    }
  }
  return polys;
};

export const GRID_POLYGONS = generatePolygons();
export const SVG_SIZE = 240;
