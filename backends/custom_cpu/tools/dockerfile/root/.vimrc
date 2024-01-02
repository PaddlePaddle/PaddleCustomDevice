set nocompatible
filetype plugin indent on

set nu
syntax enable
syntax on
set hlsearch
set incsearch
set fileencodings=utf-8,ucs-bom,gb18030,gbk,gb2312,cp936
set termencoding=utf-8
set encoding=utf-8
set cursorline
set paste
set mouse=a
set showmode
set showcmd

" expand tab to space
set expandtab
" The width of a hard tabstop measured in "spaces"
set tabstop=4
" The size of an "indent"
set shiftwidth=4
" insert a combination of spaces to simulate tab stops
set softtabstop=4

"remember last update or view postion"
 " Only do this part when compiled with support for autocommands
 if has("autocmd")
 " In text files, always limit the width of text to 78 characters
 autocmd BufRead *.txt set tw=78
 " When editing a file, always jump to the last cursor position
 autocmd BufReadPost *
 \ if line("'\"") > 0 && line ("'\"") <= line("$") |
 \ exe "normal g'\"" |
 \ endif
 endif
