load PepFus_memb.pdb

remove solvent

select pep, resid 1-62 and chain A
select pep_fus, resid 1-29 and chain A
select pep_flag, resid 30-39 and chain A
select pep_bax, resid 40-61 and chain A

select membrane, chain B

set_color c_pep_fus, [5, 11, 181]
set_color c_pep_flag, [226, 47, 47]
set_color c_pep_bax, [30, 142, 0]
set_color c_membrane, [237, 239, 208]

bg_color white

hide everything
show cartoon, pep
set cartoon_color, c_protein, protein
set cartoon_color, c_pep_fus, pep_fus
set cartoon_color, c_pep_flag, pep_flag
set cartoon_color, c_pep_bax, pep_bax

alter 30-39/, ss='H'
rebuild

show surface, membrane
set surface_color, c_membrane, membrane
set transparency, 0.5


# get_view (in pymol)

set_view (\
    -0.002266227,    0.030200798,    0.999493122,\
     0.999951184,   -0.004464095,    0.002401356,\
     0.004542065,    0.999471068,   -0.030182187,\
     0.000123171,   -0.000709161, -392.105834961,\
     1.449139357,   -1.172723651,    5.623278618,\
   260.113952637,  524.105651855,  -20.000000000 )

set depth_cue, 0
set antialias, 2

#set light, [ -0.55, -0.70, 0.15 ]
set opaque_background, off
set ray_trace_mode, 0
set ray_shadows, 0
#set ray_trace_gain, 0.05
ray 1200,900
#ray 2400,1800
png PepFus_memb.png, dpi=300

