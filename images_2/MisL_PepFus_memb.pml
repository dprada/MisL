load MisL_PepFus_memb.pdb

remove solvent

select protein, chain A
select pep_fus, res 1-29 and chain A
select pep_flag, res 30-39 and chain A
select pep_bax, res 40-61 and chain A
select pep_ompt, res 62-67 and chain A

select membrane, chain B+C

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
     0.911926866,    0.003280830,   -0.410223156,\
     0.410235047,    0.000825358,    0.911951721,\
     0.003336197,   -0.999939680,   -0.000592547,\
    -0.000001892,   -0.004228346, -400.774108887,\
    -1.744883180,   -4.076625824,  -38.864429474,\
   210.535079956,  591.197692871,   20.000000000 )

set depth_cue, 0
set antialias, 2
set orthoscopic, on

set opaque_background, off
set ray_trace_mode, 0
set ray_shadows, 0
ray 1200,900
png MisL_PepFus_memb.png, dpi=300

