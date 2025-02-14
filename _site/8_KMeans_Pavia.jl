### A Pluto.jl notebook ###
# v0.19.46

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

# ╔═╡ a0b06cc4-0b8e-4570-be05-fb19a07daf12
import Pkg; Pkg.activate(@__DIR__)

# ╔═╡ 54725f07-f7f3-429f-8773-0d4c436e8ea9
using CairoMakie, LinearAlgebra, Colors, PlutoUI, Glob, FileIO, ArnoldiMethod, CacheVariables, Clustering, ProgressLogging, Dates, SparseArrays, Random, Logging, MAT

# ╔═╡ 3763c859-1b9f-4e14-ab16-f98cabf01b61
html"""<style>
main {
    max-width: 66%;
    margin-left: 1%;
    margin-right: 2% !important;
}
"""

# ╔═╡ fa7367f5-b991-42d3-8839-31456a9208fe
@bind Location Select(["Pavia", "PaviaUni",])

# ╔═╡ 3eef6ce3-d930-4632-acfb-a5fa46c3e75e
filepath = joinpath(@__DIR__, "MAT Files", "$Location.mat")

# ╔═╡ 7dd49ca7-af66-409f-b10c-5ea453918ac2
gt_filepath = joinpath(@__DIR__, "GT Files", "$Location.mat")

# ╔═╡ 251613f2-a438-4f61-8dd5-77e2df014925
CACHEDIR = joinpath(@__DIR__, "cache_files", "Aerial Datasets")

# ╔═╡ c561b52c-9464-4b8c-a2c4-7529a261cdbd
function cachet(@nospecialize(f), path)
	whenrun, timed_results = cache(path) do
		return now(), @timed f()
	end
	@info "Was run at $whenrun (runtime = $(timed_results.time) seconds)"
	timed_results.value
end

# ╔═╡ eaf53671-1621-4231-b373-67fa178e04b8
vars = matread(filepath)

# ╔═╡ c1629b80-612c-4c4b-a6e2-4255f3b34dd8
vars_gt = matread(gt_filepath)

# ╔═╡ 8ed914df-317c-48ba-98da-1416d4ea16dc
loc_dict_keys = Dict(
	"Pavia" => ("pavia", "pavia_gt"),
	"PaviaUni" => ("paviaU", "paviaU_gt")
)

# ╔═╡ 11ddd2bf-5932-4927-bdf7-6bc156663917
data_key, gt_key = loc_dict_keys[Location]

# ╔═╡ 3f3329c2-e322-4dfb-9e22-7f16d5a3bd84
data = vars[data_key]

# ╔═╡ 8306a5ea-ef0a-4ec0-8a2d-920c8362d2db
gt_data = vars_gt[gt_key]

# ╔═╡ 66bf2550-47eb-49e3-a664-e9dc07320ca4
gt_labels = sort(unique(gt_data))

# ╔═╡ 5ead6543-4eea-47e7-8245-25f98273e058
bg_indices = findall(gt_data .== 0)

# ╔═╡ Cell order:
# ╠═3763c859-1b9f-4e14-ab16-f98cabf01b61
# ╠═a0b06cc4-0b8e-4570-be05-fb19a07daf12
# ╠═54725f07-f7f3-429f-8773-0d4c436e8ea9
# ╠═fa7367f5-b991-42d3-8839-31456a9208fe
# ╠═3eef6ce3-d930-4632-acfb-a5fa46c3e75e
# ╠═7dd49ca7-af66-409f-b10c-5ea453918ac2
# ╠═251613f2-a438-4f61-8dd5-77e2df014925
# ╠═c561b52c-9464-4b8c-a2c4-7529a261cdbd
# ╠═eaf53671-1621-4231-b373-67fa178e04b8
# ╠═c1629b80-612c-4c4b-a6e2-4255f3b34dd8
# ╠═8ed914df-317c-48ba-98da-1416d4ea16dc
# ╠═11ddd2bf-5932-4927-bdf7-6bc156663917
# ╠═3f3329c2-e322-4dfb-9e22-7f16d5a3bd84
# ╠═8306a5ea-ef0a-4ec0-8a2d-920c8362d2db
# ╠═66bf2550-47eb-49e3-a664-e9dc07320ca4
# ╠═5ead6543-4eea-47e7-8245-25f98273e058
