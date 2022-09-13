let SessionLoad = 1
let s:so_save = &g:so | let s:siso_save = &g:siso | setg so=0 siso=0 | setl so=-1 siso=-1
let v:this_session=expand("<sfile>:p")
silent only
silent tabonly
cd ~/research/vrm/_vrm_kevin/PIFu-original/PIFu
if expand('%') == '' && !&modified && line('$') <= 1 && getline(1) == ''
  let s:wipebuf = bufnr('%')
endif
let s:shortmess_save = &shortmess
set shortmess=aoO
badd +34 apps/train_shape.py
badd +52 ~/research/vrm/_vrm_kevin/PIFu-original/PIFu/lib/mesh_util.py
badd +67 ~/research/vrm/_vrm_kevin/PIFu-original/PIFu/lib/net_util.py
badd +110 ~/research/vrm/_vrm_kevin/PIFu-original/PIFu/lib/options.py
badd +386 ~/research/vrm/_vrm_kevin/PIFu-original/PIFu/lib/data/TrainDataset.py
badd +69 ~/research/vrm/_vrm_kevin/PIFu-original/PIFu/lib/data/TrainDatasetVRM.py
argglobal
%argdel
edit ~/research/vrm/_vrm_kevin/PIFu-original/PIFu/lib/data/TrainDataset.py
let s:save_splitbelow = &splitbelow
let s:save_splitright = &splitright
set splitbelow splitright
wincmd _ | wincmd |
vsplit
1wincmd h
wincmd _ | wincmd |
split
1wincmd k
wincmd w
wincmd w
let &splitbelow = s:save_splitbelow
let &splitright = s:save_splitright
wincmd t
let s:save_winminheight = &winminheight
let s:save_winminwidth = &winminwidth
set winminheight=0
set winheight=1
set winminwidth=0
set winwidth=1
exe '1resize ' . ((&lines * 29 + 30) / 60)
exe 'vert 1resize ' . ((&columns * 116 + 120) / 240)
exe '2resize ' . ((&lines * 28 + 30) / 60)
exe 'vert 2resize ' . ((&columns * 116 + 120) / 240)
exe 'vert 3resize ' . ((&columns * 123 + 120) / 240)
argglobal
balt ~/research/vrm/_vrm_kevin/PIFu-original/PIFu/lib/data/TrainDatasetVRM.py
setlocal fdm=manual
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal fen
silent! normal! zE
let &fdl = &fdl
let s:l = 374 - ((13 * winheight(0) + 14) / 29)
if s:l < 1 | let s:l = 1 | endif
keepjumps exe s:l
normal! zt
keepjumps 374
normal! 09|
wincmd w
argglobal
if bufexists(fnamemodify("~/research/vrm/_vrm_kevin/PIFu-original/PIFu/lib/data/TrainDatasetVRM.py", ":p")) | buffer ~/research/vrm/_vrm_kevin/PIFu-original/PIFu/lib/data/TrainDatasetVRM.py | else | edit ~/research/vrm/_vrm_kevin/PIFu-original/PIFu/lib/data/TrainDatasetVRM.py | endif
if &buftype ==# 'terminal'
  silent file ~/research/vrm/_vrm_kevin/PIFu-original/PIFu/lib/data/TrainDatasetVRM.py
endif
balt ~/research/vrm/_vrm_kevin/PIFu-original/PIFu/lib/data/TrainDataset.py
setlocal fdm=manual
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal fen
silent! normal! zE
let &fdl = &fdl
let s:l = 71 - ((10 * winheight(0) + 14) / 28)
if s:l < 1 | let s:l = 1 | endif
keepjumps exe s:l
normal! zt
keepjumps 71
normal! 033|
wincmd w
argglobal
if bufexists(fnamemodify("~/research/vrm/_vrm_kevin/PIFu-original/PIFu/lib/mesh_util.py", ":p")) | buffer ~/research/vrm/_vrm_kevin/PIFu-original/PIFu/lib/mesh_util.py | else | edit ~/research/vrm/_vrm_kevin/PIFu-original/PIFu/lib/mesh_util.py | endif
if &buftype ==# 'terminal'
  silent file ~/research/vrm/_vrm_kevin/PIFu-original/PIFu/lib/mesh_util.py
endif
balt apps/train_shape.py
setlocal fdm=manual
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal fen
silent! normal! zE
let &fdl = &fdl
let s:l = 16 - ((15 * winheight(0) + 29) / 58)
if s:l < 1 | let s:l = 1 | endif
keepjumps exe s:l
normal! zt
keepjumps 16
normal! 016|
wincmd w
2wincmd w
exe '1resize ' . ((&lines * 29 + 30) / 60)
exe 'vert 1resize ' . ((&columns * 116 + 120) / 240)
exe '2resize ' . ((&lines * 28 + 30) / 60)
exe 'vert 2resize ' . ((&columns * 116 + 120) / 240)
exe 'vert 3resize ' . ((&columns * 123 + 120) / 240)
tabnext 1
if exists('s:wipebuf') && len(win_findbuf(s:wipebuf)) == 0 && getbufvar(s:wipebuf, '&buftype') isnot# 'terminal'
  silent exe 'bwipe ' . s:wipebuf
endif
unlet! s:wipebuf
set winheight=1 winwidth=20
let &shortmess = s:shortmess_save
let &winminheight = s:save_winminheight
let &winminwidth = s:save_winminwidth
let s:sx = expand("<sfile>:p:r")."x.vim"
if filereadable(s:sx)
  exe "source " . fnameescape(s:sx)
endif
let &g:so = s:so_save | let &g:siso = s:siso_save
set hlsearch
let g:this_session = v:this_session
let g:this_obsession = v:this_session
doautoall SessionLoadPost
unlet SessionLoad
" vim: set ft=vim :
