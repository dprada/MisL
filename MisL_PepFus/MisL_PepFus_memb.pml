load MisL_PepFus_memb.pdb

remove solvent

select protein, chain A
select pep_fus, res 1-29 and chain A
select pep_flag, res 30-39 and chain A
select pep_bax, res 40-61 and chain A
select pep_ompt, res 62-67 and chain A

select membrane, chain B

set_color c_protein, [255,165,0]
set_color c_pep_fus, [5, 11, 181]
set_color c_pep_flag, [226, 47, 47]
set_color c_pep_bax, [30, 142, 0]
set_color c_pep_ompt, [28, 28, 28]
set_color c_membrane, [237, 239, 208]

bg_color white

hide everything
show cartoon, protein
set cartoon_color, c_protein, protein
set cartoon_color, c_pep_fus, pep_fus
set cartoon_color, c_pep_flag, pep_flag
set cartoon_color, c_pep_bax, pep_bax
set cartoon_color, c_pep_ompt, pep_ompt

alter 30-39/, ss='H'
rebuild

show surface, membrane
set surface_color, c_membrane, membrane
set transparency, 0.5


# get_view (in pymol)

set_view (\
     0.999225378,    0.005941777,    0.037773095,\
     0.038228478,   -0.129932165,   -0.990764797,\
    -0.000974112,    0.991458416,   -0.130063429,\
    -0.000035912,   -0.002369836, -385.337371826,\
     2.146217823,   -1.144892812,   28.743066788,\
   253.344345093,  517.335937500,  -20.000000000 )

set depth_cue, 0
set antialias, 2

set opaque_background, off
set ray_trace_mode, 0
set ray_shadows, 0
ray 1200,900
png MisL_PepFus_memb.png, dpi=300

