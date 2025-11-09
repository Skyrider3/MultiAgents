%%%%%%%%%%%%%%%%%          TP.cls       %%%%%%%%%%%%%%%%%%
%
%   Format file for Tim Perutz's articles.  
%   Various changes made by Ivan Smith (2016)
%   Based on Colin Rourke's file gtpart.cls for  
%   Geometry & Topology Publications.  
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  Check for fairly recent version of latex2e :
%
\NeedsTeXFormat{LaTeX2e}[1994/12/01]
%
\LoadClass[11pt]{article}  
\usepackage{amsthm,amssymb} 
\usepackage[leqno]{amsmath}
\usepackage{tikz}
\usetikzlibrary{arrows}
\tikzset{>=latex}
\tikzset{inner sep=3pt}






\newif\ifmicrotype
\DeclareOption{microtype}{\microtypetrue} 
\ProcessOptions\relax
\AtBeginDocument{%
  \def\tempaa{\thesection.\arabic{equation}}%
  \ifx\theequation\tempaa
     \def\theequation{\thesection\hbox{--}\arabic{equation}}%
  \fi
  \ifmicrotype
     \RequirePackage{microtype}%
     \DeclareMicrotypeAlias{mnt}{ptm}%
  \fi
  \DeclareMathSizes{10}   {10}   {7}{6}% from msp.cls
  \DeclareMathSizes{10.95}{10.95}{8}{6}% from msp.cls
}
%
%  Times fonts and Times math symbols 
%
\renewcommand{\sfdefault}{cmr}
\DeclareFontFamily{OT1}{ptm}{}
\DeclareFontShape{OT1}{ptm}{m}{n} { <-> ptmr}{}
\DeclareFontShape{OT1}{ptm}{m}{it}{ <-> ptmri}{}
\DeclareFontShape{OT1}{ptm}{m}{sl}{ <->ptmro}{}
\DeclareFontShape{OT1}{ptm}{m}{sc}{ <-> ptmrc}{}
\DeclareFontShape{OT1}{ptm}{b}{n} { <-> ptmb}{}
\DeclareFontShape{OT1}{ptm}{b}{it}{ <-> ptmbi}{}     
\DeclareFontShape{OT1}{ptm}{bx}{n} {<->ssub * ptm/b/n}{}
\DeclareFontShape{OT1}{ptm}{bx}{it}{<->ssub * ptm/b/it}{}
\def\rmdefault{ptm}

%
\DeclareFontFamily{OT1}{psy}{}      
\DeclareFontShape{OT1}{psy}{m}{n}{ <-> s * [0.9] psyr}{}
%
\DeclareFontFamily{OMS}{ptm}{}     
\DeclareFontShape{OMS}{ptm}{m}{n}{ <8> <9> <10> gen * cmsy }{}
\DeclareFontFamily{OMS}{cmtt}{}     
\DeclareFontShape{OMS}{cmtt}{m}{n}{ <8> <9> <10> gen * cmsy }{}

\SetSymbolFont{operators}{normal}{OT1}{ptm}{m}{n}   
\SetSymbolFont{operators}{bold}{OT1}{ptm}{b}{n}     
%
\DeclareSymbolFont{emsy}{OT1}{ptm}{m}{it}
\DeclareSymbolFont{emsr}{OT1}{ptm}{m}{n}
\DeclareSymbolFont{emcmr}{OT1}{cmr}{m}{n}   
\DeclareSymbolFont{emsymb}{OT1}{psy}{m}{n}  
%
\DeclareMathSymbol a{\mathalpha}{emsy}{"61}
\DeclareMathSymbol b{\mathalpha}{emsy}{"62}
\DeclareMathSymbol c{\mathalpha}{emsy}{"63}
\DeclareMathSymbol d{\mathalpha}{emsy}{"64}
\DeclareMathSymbol e{\mathalpha}{emsy}{"65}
\DeclareMathSymbol f{\mathalpha}{emsy}{"66}
\DeclareMathSymbol g{\mathalpha}{emsy}{"67}
\DeclareMathSymbol h{\mathalpha}{emsy}{"68}
\DeclareMathSymbol i{\mathalpha}{emsy}{"69}
\DeclareMathSymbol j{\mathalpha}{emsy}{"6A}
\DeclareMathSymbol k{\mathalpha}{emsy}{"6B}
\DeclareMathSymbol l{\mathalpha}{emsy}{"6C}
\DeclareMathSymbol m{\mathalpha}{emsy}{"6D}
\DeclareMathSymbol n{\mathalpha}{emsy}{"6E}
\DeclareMathSymbol o{\mathalpha}{emsy}{"6F}
\DeclareMathSymbol p{\mathalpha}{emsy}{"70}
\DeclareMathSymbol q{\mathalpha}{emsy}{"71}
\DeclareMathSymbol r{\mathalpha}{emsy}{"72}
\DeclareMathSymbol s{\mathalpha}{emsy}{"73}
\DeclareMathSymbol t{\mathalpha}{emsy}{"74}
\DeclareMathSymbol u{\mathalpha}{emsy}{"75}
\DeclareMathSymbol v{\mathalpha}{emsy}{"76}
\DeclareMathSymbol w{\mathalpha}{emsy}{"77}
\DeclareMathSymbol x{\mathalpha}{emsy}{"78}
\DeclareMathSymbol y{\mathalpha}{emsy}{"79}
\DeclareMathSymbol z{\mathalpha}{emsy}{"7A}
\DeclareMathSymbol A{\mathalpha}{emsy}{"41}
\DeclareMathSymbol B{\mathalpha}{emsy}{"42}
\DeclareMathSymbol C{\mathalpha}{emsy}{"43}
\DeclareMathSymbol D{\mathalpha}{emsy}{"44}
\DeclareMathSymbol E{\mathalpha}{emsy}{"45}
\DeclareMathSymbol F{\mathalpha}{emsy}{"46}
\DeclareMathSymbol G{\mathalpha}{emsy}{"47}
\DeclareMathSymbol H{\mathalpha}{emsy}{"48}
\DeclareMathSymbol I{\mathalpha}{emsy}{"49}
\DeclareMathSymbol J{\mathalpha}{emsy}{"4A}
\DeclareMathSymbol K{\mathalpha}{emsy}{"4B}
\DeclareMathSymbol L{\mathalpha}{emsy}{"4C}
\DeclareMathSymbol M{\mathalpha}{emsy}{"4D}
\DeclareMathSymbol N{\mathalpha}{emsy}{"4E}
\DeclareMathSymbol O{\mathalpha}{emsy}{"4F}
\DeclareMathSymbol P{\mathalpha}{emsy}{"50}
\DeclareMathSymbol Q{\mathalpha}{emsy}{"51}
\DeclareMathSymbol R{\mathalpha}{emsy}{"52}
\DeclareMathSymbol S{\mathalpha}{emsy}{"53}
\DeclareMathSymbol T{\mathalpha}{emsy}{"54}
\DeclareMathSymbol U{\mathalpha}{emsy}{"55}
\DeclareMathSymbol V{\mathalpha}{emsy}{"56}
\DeclareMathSymbol W{\mathalpha}{emsy}{"57}
\DeclareMathSymbol X{\mathalpha}{emsy}{"58}
\DeclareMathSymbol Y{\mathalpha}{emsy}{"59}
\DeclareMathSymbol Z{\mathalpha}{emsy}{"5A}
\DeclareMathSymbol{\bullet}{\mathalpha}{emsymb}{"B7}
\DeclareMathSymbol{\regis}{\mathalpha}{emsymb}{"D2}
\def\Bullet{\leavevmode\unkern{$\m@th\bullet$}\kern.32em\ignorespaces}
\def\Regis{\leavevmode\raise.5ex\hbox{$\m@th\regis$}}
\DeclareMathSymbol +{\mathbin}{emcmr}{`+}
\DeclareMathSymbol ={\mathrel}{emcmr}{`=}  
\DeclareMathSymbol{\Gamma}{\mathalpha}{emcmr}{"00}
\DeclareMathSymbol{\Delta}{\mathalpha}{emcmr}{"01}
\DeclareMathSymbol{\Theta}{\mathalpha}{emcmr}{"02}
\DeclareMathSymbol{\Lambda}{\mathalpha}{emcmr}{"03}
\DeclareMathSymbol{\Xi}{\mathalpha}{emcmr}{"04}
\DeclareMathSymbol{\Pi}{\mathalpha}{emcmr}{"05}
\DeclareMathSymbol{\Sigma}{\mathalpha}{emcmr}{"06}
\DeclareMathSymbol{\Upsilon}{\mathalpha}{emcmr}{"07}
\DeclareMathSymbol{\Phi}{\mathalpha}{emcmr}{"08}
\DeclareMathSymbol{\Psi}{\mathalpha}{emcmr}{"09}
\DeclareMathSymbol{\Omega}{\mathalpha}{emcmr}{"0A}
\DeclareMathSizes{7.6}{8}{6}{5}

%
%    some accents and useful symbols (from L'Ensiegnment class file)
%
\def\`#1{{\accent"12 #1}}            % grave accent
\def\H#1{{\accent"CD #1}}            % double acute accent
\chardef\J="11                  
\def\j{{\sf\J}}                      % j without dot 
\chardef\AA="C8                      % Angstrom
\chardef\gbp="A3                     % pound sign           
\chardef\TIL="81                     % tilde on line
\chardef\endash="B1                
\chardef\emdash="D0                
\chardef\pourmille="BD               % %sign
\chardef\aoben="E3                   % ^a
\chardef\ooben="EB                   % ^o
\def\S{\leavevmode\unkern{\char"A7}\kern.1em\ignorespaces} 
\def\D#1{{\accent"C7 #1}}            % overdot
\DeclareMathAccent{\dot}{\mathalpha}{operators}{"C7} % ditto math mode



%  hyperref stuff
%
\usepackage{hyperref}  
\hypersetup{%
  bookmarksnumbered=true,%
  bookmarks=true,%
  colorlinks=true,%
  linkcolor=blue,%
  citecolor=blue,%
  filecolor=blue,%
  menucolor=blue,%
  pagecolor=blue,%
  urlcolor=blue,%
  pdfnewwindow=true,%
  pdfstartview=FitBH}
%
%

%\makeatletter
 \newcommand{\raisedlink}[1]{\Hy@raisedlink{\hypertarget{#1}{}}}
%\makeatother


%  Basic layout :
%
%  get print centerpage:
%

\headsep 10pt %23pt
\footskip 20pt %35pt
\hoffset -16truemm
\voffset -10truemm

%
\newskip\stdskip                      % standard vertical space
\stdskip= 4pt plus3.3pt minus3.3pt    
%
\setlength{\textheight}{8.5in}          
\setlength{\textwidth}{6.2in}         
\flushbottom                           
\setlength{\parindent}{0pt}            
\setlength{\parskip}{\stdskip}
\setlength{\medskipamount}{\stdskip}
\setlength{\mathsurround}{0.8pt}     
\setlength{\labelsep}{0.75em} 
\newcommand{\stdspace}{\hskip 0.75em plus 0.15em \ignorespaces}
\let\qua\stdspace  %  useful abbreviation 
\renewcommand{\baselinestretch}{1.0}  %  

%
%  Some style commands (\ppar is for principal paragraph breaks, \sh is
%  for subheadings and \rk for remarks etc -- see also theorem style 
%  below ) :
%

\newcommand{\ppar}{\par\goodbreak\medskip}
\newcommand{\sh}[1]{\penalty-800\ppar{\bf #1}\par\medskip\nobreak}
\newcommand{\rk}[1]{\ppar{\bf #1}\stdspace}    
%
%
%   Theorem style.  There are two recommended styles :
%
%   plain : for theorems, corollaries etc with heading bold
%   and left justified, optional note bracketed in roman type
%   and statement in slanted type.
%
%   definition : (alias remark)  for definitions, remarks etc with
%   heading bold and left justified, optional note unbracketed in 
%   slanted type and statement in roman type.
%   
%
%  Redefine the amsthm styles plain, definition and remark:
%
\newtheoremstyle{plain}{13.2pt plus6.6pt minus6.6pt}{6.6pt plus3.3pt minus3.3pt}%
{\sl}{}{\bf}{}{0.75em}{\thmname{#1}\thmnumber{ #2}\thmnote{\rm\stdspace(#3)}}
%
\newtheoremstyle{definition}{13.2pt plus6.6pt minus6.6pt}{6.6pt plus3.3pt minus3.3pt}%
{\rm}{}{\bf}{}{0.75em}{\thmname{#1}\thmnumber{ #2}\thmnote{\rm\stdspace(#3)}}
%

\newtheoremstyle{remark}{13.2pt plus6.6pt minus6.6pt}{6.6pt plus3.3pt minus3.3pt}%
{\rm}{}{\bf}{}{0.75em}{\thmname{#1}\thmnumber{ #2}\thmnote{\rm\stdspace(#3)}}
%
%  Default theorem style :
%\theoremstyle{plain}               
%
%  adapt the amsthm proof environment
%
	%\renewenvironment{proof}[1][\proofname]{\par
  	%	\pushQED{\qed}%
  	%	\normalfont \topsep\stdskip
  	%	\trivlist
  	%	\item[\hskip\labelsep
       % 		\bf#1]\ignorespaces
%}{%
 % \popQED\endtrivlist\@endpefalse
%}
%                 Knuth's \square macro :
%
%\def\sqr#1#2{{\vcenter{\vbox{\hrule  height.#2pt
	%\hbox{\vrule width.#2pt height#1pt \kern#1pt \vrule width.#2pt}
	%\hrule height.#2pt}}}}
%
%\def\sq{\sqr55}           %    A small square for end-of-proofs. 
%\def\qedsymbol{$\sqr55$}  %   (Define other size squares by varing the
%                         %   the two numbers.)
%

%
%  metadata items:
%
%  title
%
\def\title{\@ifnextchar[\doubletitle\singletitle}
 \def\doubletitle[#1]#2{\def\thetitle{#2}\def\theshorttitle{#1}}
 \def\singletitle#1{\def\thetitle{#1}\def\theshorttitle{#1}}
%
%  author(s)
%
\def\author{\@ifnextchar[\doubleauthor\singleauthor}
\def\doubleauthor[#1]#2{\edef\previousauthors{\theauthors}
 \ifx\theauthors\relax\def\theauthors{#2}\else
 \def\theauthors{\previousauthors\par#2}\fi
 \edef\previousrootauthors{\rootauthors}
 \ifx\rootauthors\relax
     \ifx\lastauthor\relax\def\lastauthor{#1}\def\theshortauthors{#1}
     \else\edef\rootauthors{\lastauthor}\def\lastauthor{#1}
          \def\theshortauthors{\rootauthors\ and #1}\fi
 \else\edef\rootauthors{\previousrootauthors, \lastauthor}
      \def\lastauthor{#1}\def\theshortauthors{\rootauthors\ and #1}\fi}
\def\singleauthor#1{\doubleauthor[#1]{#1}}
%
%  address
%
\def\address#1{{\let\\\relax\let\newline\par\xdef\previousaddresses{\theaddress}}
 \ifx\theaddress\relax\def\theaddress{#1}\else
 \def\theaddress{\previousaddresses\par\vskip 3pt minus 2pt\par#1}\fi}
%
%  email
%
\def\email#1{{\let\_\relax\let~\relax\let\href\relax
              \xdef\previousemails{\theemail}}%
            \def\tempab{}\def\tempba{#1}\ifx\tempab\tempba\else
            \ifx\theemail\tempab{}\def\theemail{\href{mailto:#1}{#1}}\else
            \gdef\theemail{\previousemails, \href{mailto:#1}{#1}}\fi\fi}
%
%  url    %%% needs linking
%
\def\urladdr#1{{\let\_\relax\let~\relax\let\href\relax\xdef\previousurls{\theurl}}
            \def\tempab{}\def\tempba{#1}\ifx\tempab\tempba\else
            \ifx\theurl\tempab\def\theurl{\href{#1}{#1}}\else
            \def\theurl{\previousurls, \href{#1}{#1}}\fi\fi}
%
%  main abstract
%
\long\def\abstract#1\end%
{\long\gdef\theabstract{#1}\end}
\let\endabstract\relax
%
%  set metadata items to \relax to initialise macros :
%
\let\thetitle\relax\let\theauthors\relax
\let\theaddress\relax\let\theemail\relax\let\theurl\relax
\let\theabstract\relax\let\theshorttitle\relax
\let\theshortauthors\relax\let\rootauthors\relax
\let\lastauthor\relax
\let\thededicatory\relax
\def\theemail{}\def\theurl{}
%

\def\startpage#1{\relax}
\def\endpage#1{\relax}


\def\thestartpage{1}

% start of definition of \maketitle
%
\long\def\maketitle{
\count0=\thestartpage\ 
\vglue 0pt plus 3pt minus 3pt %% top space

%
% title :
%
{\parskip=0pt\leftskip 0pt plus 1fil{\def\\{\par\smallskip}\Large
\bf\thetitle}\par\vskip 3pt}  

\smallskip
%
% authors :
%
{\parskip=0pt\leftskip 0pt plus 1fil{\sc\theauthors}
\par}

\smallskip
 
\vglue 0pt plus 3pt minus 3pt
%
% abstract :
% 
{\small\leftskip 25pt\rightskip 25pt\parskip 4pt plus 2pt
\theabstract
   \ifx\thededicatory\relax\else
  \vglue 7ptplus 2pt\it\thededicatory\fi\par\smallskip}}


\def\@oddhead
{{\small\it
\ifnum\count0=\thestartpage 

\else

\ifodd\count0
\def\\{\unskip\space}
\ifx\theshorttitle\relax \thetitle \else\theshorttitle\fi
\hfill
{\bf\number\count0}\else{\bf\number\count0}
\hfill
\ifx\theshortauthors\relax \theauthors\else\theshortauthors \fi
\fi
\fi}}
\let\@evenhead\@oddhead
\def\@oddfoot{{}} 
\let\@evenfoot\@oddfoot

%
%   hacks to get various items of style correct :
%
%   set footnotes in 10pt type:
%
\let\@footnote@\footnote
\def\footnote#1{\@footnote@{\small #1}}

%
%  set captions in 10pt type  (hack of excerpt from hyperref.sty) :
%
\long\def\@caption#1[#2]#3{%
  \hyper@makecurrent{\@captype}%
  \par\addcontentsline{\csname ext@#1\endcsname}{#1}{%
    \protect\numberline{\csname the#1\endcsname}{\ignorespaces #2}%
  }%
  \begingroup
    \@parboxrestore
    \if@minipage
      \@setminipage
    \fi
    \small
    \@makecaption{\csname fnum@#1\endcsname}{%
      \ignorespaces
      \ifHy@nesting
        \hyper@@anchor{\@currentHref}{#3}%
      \else
        \Hy@raisedlink{\hyper@@anchor{\@currentHref}{\relax}}#3%
      \fi
    }%
    \par
  \endgroup
}
%
%
%  set displayskips to correct values :
%
\AtBeginDocument{\setlength{\abovedisplayskip}{\stdskip}
\setlength{\belowdisplayskip}{\stdskip}}  
%
%  get the biblio style correct (10pt with small gaps):
%

\let\@thebibliography@\thebibliography
\def\thebibliography#1 {\@thebibliography@{999}\small\parskip0pt 
plus2pt\relax\addcontentsline{toc}{section}{Bibliography}}

%
%
%  Get item spacing reasonable :
%
\let\@itemize@\itemize
\def\itemize{\@itemize@\parskip 0pt\relax}
\def\@listi{\leftmargin28.5pt\parsep 0pt\topsep 0pt 
 \itemsep4pt plus3pt minus2pt}
\let\@listI\@listi
\@listi
%
%  get enumeration labels to use \rm
%
\renewcommand{\labelenumi}{{\rm (\theenumi)}}
\renewcommand{\labelenumii}{{\rm (\theenumii)}}
\renewcommand{\labelenumiii}{{\rm (\theenumiii)}}

%  and spacing to match \itemize 
%
\let\@enumerate@\enumerate
\def\enumerate{\@enumerate@\parskip 0pt\relax}
%
%  misc definitions for comptibility with amsthm, msp.cls etc
%
\def\bysame{\leavevmode\hbox to3em{\hrulefill}\thinspace}
\def\figdir{figs}
\def\makeop#1{\expandafter\def\csname #1\endcsname{\mathop{\mathrm{#1}}\nolimits}}
\let\wtilde\widetilde
\let\wwtilde\widetilde
\let\what\widehat
\let\wwhat\widehat
\let\wbar\bar
\let\wwbar\overline
\let\proved\qedhere



\let\disablesubscriptcorrection\relax
\newenvironment{nopartheorem}{}{}
\def\half{\tfrac12}
%
\def\eqalign#1{%
 \null\,\vcenter{\openup\jot\m@th
  \ialign{\strut\hfil$\displaystyle{##}$&$\displaystyle{{}##}$\hfil
      \crcr#1\crcr}}\,}
\def\eqalignbot#1{%
 \null\,\vbox{\openup\jot\m@th
  \ialign{\strut\hfil$\displaystyle{##}$&$\displaystyle{{}##}$\hfil
      \crcr#1\crcr}}\,}
\def\eqaligntop#1{%
 \null\,\vtop{\openup\jot\m@th
  \ialign{\strut\hfil$\displaystyle{##}$&$\displaystyle{{}##}$\hfil
      \crcr#1\crcr}}\,}
\def\eqalignn#1{%
 \null\,\vcenter{\openup\jot\m@th
  \ialign{\strut\hfil$\displaystyle{##}$&$\displaystyle{{}##}$\hfil
      &\hfil$\displaystyle{##}$&$\displaystyle{{}##}$\hfil\crcr#1\crcr}}\,}

%




%
%  endgame
%

\AtEndDocument{{\small\parskip 0pt\vskip11pt minus 5pt\relax
{\sl \def\\{\futurelet\next\nocommawithnl}\def\nocommawithnl
  {\ifx\next\newline\else\unskip,\space\ignorespaces\fi}
  \theaddress\par}
{\rightskip0pt plus .4\hsize
{\def\tempab{}\tt\def~{\lower3.5pt\hbox{\char'176}}\def\_{\char'137}%
\ifx\theemail\tempab\else
  \vskip5pt minus 3pt\theemail\par\fi
  \ifx\theurl\tempab\else
  \vskip5pt minus 3pt\theurl\par\fi}
  \vskip11pt minus 5pt
  }}}
  
  
 \usepackage[all]{xy}
%\usepackage{graphicx}
\usepackage{pinlabel}
\usepackage{enumerate}
\usepackage{framed}
\usepackage{datetime}


%Macros etc.







  
\endinput
%

