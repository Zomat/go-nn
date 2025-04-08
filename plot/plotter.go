package plot

import (
	"image/color"
	"log"

	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
	"gonum.org/v1/plot/vg/draw"
)

func SavePlot(points []plotter.XYs, mainTitle string, xLabel string, yLabel string, legend []string, filename string) {
	p := plot.New()

	p.Title.Text = mainTitle
	p.X.Label.Text = xLabel
	p.Y.Label.Text = yLabel

	for i, pts := range points {
		l, err := plotter.NewScatter(pts)
		if err != nil {
			panic(err)
		}
		
		l.GlyphStyle.Shape = draw.CircleGlyph{}
		l.GlyphStyle.Radius = vg.Points(2)
		l.GlyphStyle.Color = color.RGBA{
			R: uint8((100*(i+1)) % 255),
			G: uint8((50 * i) % 255),
			B: uint8((100 * i) % 255),
			A: 255,
		}

		p.Legend.Top = true
		p.Legend.Add(legend[i], l)
		p.Add(l)
	}

	if err := p.Save(8*100, 6*100, filename); err != nil {
		log.Fatal(err)
	}
}
