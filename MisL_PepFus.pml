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

# get_view (in pymol)

set_view (\
     0.995490193,    0.079611339,    0.051051646,\
     0.023577413,    0.313836575,   -0.949165761,\
    -0.091582760,    0.946102798,    0.310549676,\
     0.000008410,   -0.000625528, -363.902343750,\
     2.454076529,   -1.028667450,   29.198543549,\
   231.904296875,  495.896118164,  -20.000000000 )

set depth_cue, 0
set antialias, 2

set opaque_background, off
set ray_trace_mode, 0
set ray_shadows, 0
ray 1200,900
png MisL_PepFus.png, dpi=300
