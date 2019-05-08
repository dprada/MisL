load PepFus.pdb

remove solvent

select pep, resid 1-62
select pep_fus, resid 1-29
select pep_flag, resid 30-39
select pep_bax, resid 40-61

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
show surface, membrane
set surface_color, c_membrane, membrane
set transparency, 0.5


# get_view (in pymol)

set_view (\
    -0.478747457,    0.383721948,   -0.789638340,\
     0.605185091,    0.795822322,    0.019805020,\
     0.636018813,   -0.468412519,   -0.613222837,\
     0.000000000,    0.000000000, -119.990379333,\
    65.150489807,   65.209884644,   31.942541122,\
    94.601325989,  145.379425049,  -20.000000000 )

set depth_cue, 0
set antialias, 2

set opaque_background, off
set ray_trace_mode, 0
set ray_shadows, 0
ray 1200,900
#ray 2400,1800
png PepFus.png, dpi=300

