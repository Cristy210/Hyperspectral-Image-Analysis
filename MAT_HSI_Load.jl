### A Pluto.jl notebook ###
# v0.19.40

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ 2630714a-458d-41dc-9727-d01de1d6d20b
import Pkg; Pkg.activate(@__DIR__)

# ╔═╡ d2349f0e-4114-476c-a58a-3c71e84dbfe7
using CairoMakie, LinearAlgebra, Colors, PlutoUI, ArchGDAL, Glob, FileIO, Interpolations, PythonCall, PaddedViews, MAT

# ╔═╡ 481e4100-84b8-11ef-1fe5-23d3b32b5761
html"""<style>
main {
    max-width: 76%;
    margin-left: 1%;
    margin-right: 2% !important;
}
"""

# ╔═╡ 020c0cab-f167-4612-be10-9adaf80c47eb
md"""
## Loading and Pre-processing Data
"""

# ╔═╡ d7407c3a-5555-4ed7-a7ad-beec4b3685a5
@bind region Select(["barbara"])

# ╔═╡ 4c54b69d-e237-4659-99b3-e82b2cd8c932
data_path_T1 = joinpath(@__DIR__, "MAT Files", region, "barbara_2013.mat")

# ╔═╡ 23d7afac-3f98-47a4-a089-9d8f65ca5410
file = matopen(data_path_T1)

# ╔═╡ 39b7d9ba-76a1-4bb2-b54f-cc4e5679fafb
D1 = read(file)

# ╔═╡ f2f2d7b5-7978-43d1-a049-19c04367e85d
sb_t1 = D1["HypeRvieW"]

# ╔═╡ 8367623f-0a7f-40a3-9bbd-8a31778ac166
@bind Band PlutoUI.Slider(1:size(sb_t1, 3), show_value=true)

# ╔═╡ 193c4aca-6be5-47be-948b-b3df3c930f7e
with_theme() do
	fig = Figure(; size=(800, 600))
	ax = Axis(fig[1, 1], aspect = DataAspect(), yreversed=true)
	image!(ax, sb_t1[:, :, Band])
	fig
end

# ╔═╡ 673361bb-0be1-4c78-90cb-88bcd7204934
data_mat = permutedims(reshape(sb_t1, :, size(sb_t1, 3)))

# ╔═╡ 64a7e9f4-b1ec-4302-b44e-24dc1ac332df
col_norms = [norm(data_mat[:, i]) for i in 1:size(data_mat, 2)]

# ╔═╡ 08ef8b39-034c-471d-816d-aff08b10b18b
Norm_vec = [data_mat[:, i] ./ col_norms[i] for i in 1:size(data_mat, 2)]

# ╔═╡ 2bd01a7f-7722-41de-aa43-320d789c5e05
Norm_mat = hcat(Norm_vec...)

# ╔═╡ 166febef-0289-4515-ab3f-799b8b73aae9


# ╔═╡ Cell order:
# ╟─481e4100-84b8-11ef-1fe5-23d3b32b5761
# ╟─020c0cab-f167-4612-be10-9adaf80c47eb
# ╠═2630714a-458d-41dc-9727-d01de1d6d20b
# ╠═d2349f0e-4114-476c-a58a-3c71e84dbfe7
# ╠═d7407c3a-5555-4ed7-a7ad-beec4b3685a5
# ╠═4c54b69d-e237-4659-99b3-e82b2cd8c932
# ╠═23d7afac-3f98-47a4-a089-9d8f65ca5410
# ╠═39b7d9ba-76a1-4bb2-b54f-cc4e5679fafb
# ╠═f2f2d7b5-7978-43d1-a049-19c04367e85d
# ╠═8367623f-0a7f-40a3-9bbd-8a31778ac166
# ╠═193c4aca-6be5-47be-948b-b3df3c930f7e
# ╠═673361bb-0be1-4c78-90cb-88bcd7204934
# ╠═64a7e9f4-b1ec-4302-b44e-24dc1ac332df
# ╠═08ef8b39-034c-471d-816d-aff08b10b18b
# ╠═2bd01a7f-7722-41de-aa43-320d789c5e05
# ╠═166febef-0289-4515-ab3f-799b8b73aae9
