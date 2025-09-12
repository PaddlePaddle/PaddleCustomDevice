# .bashrc

# Source global definitions
if [ -f /etc/bashrc ]; then
    . /etc/bashrc
fi

# no auto logout
export TMOUT=0

# Locales
export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8
export LANGUAGE=en_US.UTF-8

# Aliases

alias rm='rm -i'
alias cp='cp -i'
alias mv='mv -i'

alias l='ls -lF'
alias ll='ls -alF'
alias lt='ls -ltrF'
alias ll='ls -alF'
alias lls='ls -alSrF'
alias llt='ls -altrF'

# Colorize directory listing
alias ls="ls -p --color=auto"
alias pstall='pip install -U --no-deps --force-reinstall'

# Colorize grep
if echo hello|grep --color=auto l >/dev/null 2>&1; then
  alias grep='grep --color=always'
  export GREP_COLOR="1;31"
fi

# Shell
export CLICOLOR="1"

source ~/.scripts/git-prompt.sh
export PS1="\[\e[1;33m\]λ\[\e[0m\] \h \[\e[1;32m\]\w\[\e[1;33m\]\$(__git_ps1 \" \[\e[35m\]{\[\e[36m\]%s\[\e[35m\]}\") \[\e[0m\]"
source ~/.scripts/git-completion.sh
