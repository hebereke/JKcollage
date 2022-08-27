import os
import math
import PIL.Image
import PIL.ImageDraw
import PIL.ImageFont
import glob
import random
from pulp import *

class JK(PIL.Image.Image):
    def __init__(self, p):
        self.w = p.width
        self.h = p.height
        self.x = 0.0
        self.y = 0.0
        self.p = p
        self.orig_w = self.w
        self.orig_h = self.h
        self.orig_p = self.p
        self.type()
    def type(self, tol=0.1):
        if (2*self.w - self.h) < tol*self.h:
            self.type = 1
        elif (2*self.h - self.w) < tol*self.w:
            self.type = 2
        else:
            self.type = 3
    def area(self):
        return self.w * self.h
    def resize(self):
        self.p = self.p.resize((self.w, self.h), PIL.Image.LANCZOS)
        return True

def norm(pics, W, area_tol=0.0, keep_aspect=False, reserved_area=0.0):
    expect_area = W ** 2 - reserved_area
    if expect_area < 0:
        raise ValueError('reserved_area is too much large')
    num = 3 * [0]
    for p in pics:
        num[p.type-1] += 1
    desired_width = math.sqrt(expect_area / (2 * (num[0] + num[1] + 2 * num[2])))
    unit_width = math.floor(math.sqrt(1-area_tol) * desired_width)
    for p in pics:
        if p.type == 1:
            ratio = unit_width / p.w
            p.w = unit_width
            if keep_aspect:
                p.h = math.floor(ratio * p.h)
            else:
                p.h = 2 * unit_width
        elif p.type == 2:
            ratio = unit_width / p.h
            if keep_aspect:
                p.w = math.floor(ratio * p.w)
            else:
                p.w = 2 * unit_width
            p.h = unit_width
        elif p.type == 3:
            ratio = 2 * unit_width / max(p.w, p.h)
            if keep_aspect:
                p.w = math.floor(ratio * p.w)
                p.h = math.floor(ratio * p.h)
            else:
                p.w = 2 * unit_width
                p.h = 2 * unit_width
        p.resize()
    return unit_width

## PuLP base solver
def PuLP_method(pics, W):
    UB = W
    N = len(pics)
    w = [p.w for p in pics]
    h = [p.h for p in pics]
    x = [ LpVariable('x{:03d}'.format(i), lowBound = 0)\
        for i in range(N)]
    y = [ LpVariable('y{:03d}'.format(i), lowBound = 0)\
        for i in range(N)]
    u = [[LpVariable('u{:03d}{:03d}'.format(i,j), cat = LpBinary )\
        for j in range(N)]\
            for i in range(N)]
    v = [[LpVariable('v{:03d}{:03d}'.format(i,j), cat = LpBinary )\
        for j in range(N)]\
            for i in range(N)]
    H = LpVariable('H', cat = 'Integer')
    m = LpProblem(sense = LpMinimize)
    m += H
    for i in range(N):
        for j in range(N):
            m += x[i] + w[i] <= x[j] + W * (1-u[i][j])
            m += y[i] + h[i] <= y[j] + UB * (1-v[i][j])
            if i < j:
                m += u[i][j] + u[j][i] + v[i][j] + v[j][i] >= 1
    for i in range(N):
        m += x[i] <= W-w[i]
        m += y[i] <= H-h[i]
    m.solve ()
    for i in range(N):
        pics[i].x = value(x[i])
        pics[i].y = value(y[i])
    return pics, int(value(H))

## BL base solver
def bl_candidates(i, x, y, w, h):
    cand = [(0 ,0)]
    for j in range(i):
        for k in range(j):
            cand += [(x[j] + w[j], y[k] + h[k])]
            cand += [(x[k] + w[k], y[j] + h[j])]
    for j in range(i):
        cand += [(0, y[j] + h[j])]
        cand += [(x[j] + w[j], 0)]
    return cand

def is_feas (i, p, x, y, w, h, W):
    if (p[0] < 0) or (W < p[0] + w[i]):
        return False
    for j in range(i):
        if max(p[0], x[j]) < min(p[0] + w[i], x[j] + w[j]):
            if max(p[1], y[j]) < min(p[1] + h[i], y[j] + h[j]):
                return False
    return True

def BL_method(pics,W):
    w = [p.w for p in pics]
    h = [p.h for p in pics]
    x, y = [], []
    for i in range(len(w)):
        blfp = []
        cand = bl_candidates(i, x, y, w, h)
        for p in cand:
            if is_feas (i, p, x, y, w, h, W):
                blfp += [p]
        min_p = min(blfp, key = lambda v:(v[1], v[0]))
        x += [min_p [0]]
        y += [min_p [1]]
    H = 0
    for i in range(len(pics)):
        pics[i].x = x[i]
        pics[i].y = y[i]
        H = max(H, pics[i].y + pics[i].h)
    return pics, H

def draw_result(pics, H, W, output,\
    bg = 'white',\
    display_result = False, frame_only = False, frame_width=0,\
    font_size=12, font_name='C:\Windows\Fonts\scriptbl.ttf'):
    im = PIL.Image.new('RGB', (W, H), bg)
    draw = PIL.ImageDraw.Draw(im)
    font = PIL.ImageFont.truetype(font_name, font_size)
    for i in range(len(pics)):
        p = pics[i]
        x1 = p.x
        y1 = p.y
        x2 = p.x + p.w
        y2 = p.y + p.h
        if frame_only:
            if p.type == 1:
                fc = 'blue'
            elif p.type == 2:
                fc = 'yellow'
            else:
                fc = 'red'
            title = '{}'.format(i)
            text_box_width = len(title) * font_size
            draw.rectangle((x1, y1, x2, y2), \
                fill = fc, outline = 'black', width=frame_width)
            draw.text((x1 + font_size/2.0, y1 + font_size/2.0), title, \
                fill = 'black', font = font, align = 'center')
        else:
            im.paste(p.p, (p.x, p.y))
    im.save(output)
    if display_result:
        im.show()

def title(text, font, size):
    font = PIL.ImageFont.truetype(font, size)
    # text box width and height
    im0 = PIL.Image.new('RGB', (0, 0), 'white')
    draw0 = PIL.ImageDraw.Draw(im0)
    text_width, text_height = draw0.textsize(text, font = font)
    # title box
    box_width = int(math.floor(1.5 * text_width))
    box_height = int(math.floor(1.5 * text_height))
    im = PIL.Image.new('RGB', (box_width, box_height), 'white')
    draw = PIL.ImageDraw.Draw(im)
    draw.rectangle((0, 0, box_width, box_height), \
        fill = 'white', outline = 'black', width=2)
    text_width, text_height = draw.textsize(text, font = font)
    draw.text(((box_width - text_width)/2.0, (box_height - text_height)/2.0),\
        text = text, fill = 'black', font = font)
    return im

if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser(prog = 'JKcollage',
        description='Create CD jacket collage image',
        formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--width', '-w', dest = 'width',\
        help = 'image width', type = int, default = 500, metavar = 'NUM')
    parser.add_argument('--area_tol',\
        help = 'area tollerance', default = 0.0, metavar = 'NUM')
    parser.add_argument('--image_num',\
        help = 'number of images to be used', type = int, default = None, metavar = 'NUM')
    parser.add_argument('--output',\
        help = 'output image path', default = 'JKcollage.jpg', metavar = 'FILE')
    parser.add_argument('--recursive', '-r',\
        help = 'search CD jacket recursively', action = 'store_true')
    parser.add_argument('--keep_aspect',\
        help = 'keep original image aspect ratio', action = 'store_true')
    parser.add_argument('--method',\
        help = 'specify packing method [PuLP, BL]', default = 'BL')
    parser.add_argument('--frame_only',\
        help = 'output frame only', action = 'store_true')
    parser.add_argument('--display_result',\
        help = 'display result', action = 'store_true')
    parser.add_argument('--background_color',\
        help = 'background color', default = 'white')
    parser.add_argument('--title',\
        help = 'add title', action = 'store_true')
    parser.add_argument('--title_text',\
        help = 'title text', default = 'JKcollage')
    parser.add_argument('--title_size',\
        help = 'title font size', type = int, default = 12)
    parser.add_argument('--title_font',\
        help = 'title font', default = 'C:\\Windows\\Fonts\\scriptbl.ttf')
    parser.add_argument('folders',\
        help = 'number of images to be used', nargs = '+')
    args = parser.parse_args()
    pics = []
    for f in args.folders:
        pics = pics + [JK(PIL.Image.open(f)) for f in glob.glob(os.path.join(f, '*.jpg'), recursive = args.recursive)]
    if args.image_num is not None:
        pics = [pics[i] for i in range(args.image_num)]
    if args.title:
        title_im = title(args.title_text, args.title_font, args.title_size)
        title_area = title_im.width * title_im.height
    else:
        title_area = 0.0
    uw = norm(pics, args.width, area_tol = args.area_tol,\
        keep_aspect = args.keep_aspect, reserved_area = title_area)
    if args.title:
        title_order = random.choice([i for i in range(len(pics))])
        orig_pics = pics
        pics = []
        for i in range(len(orig_pics)):
            if i==title_order:
                pics.append(JK(title_im))
            pics.append(orig_pics[i])
    if args.method == 'PuLP':
        pics, H = PuLP_method(pics, W)
    elif args.method == 'BL':
        pics, height = BL_method(pics, args.width)
    draw_result(pics, height, args.width, args.output,\
        display_result = args.display_result,\
        bg = args.background_color,\
        font_size = min(24, int(math.floor(0.6*uw))),\
        font_name = args.title_font,\
        frame_only = args.frame_only)