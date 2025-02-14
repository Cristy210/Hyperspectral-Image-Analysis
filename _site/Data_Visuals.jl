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

# ╔═╡ 6750f0bf-429c-407c-a8fd-0ea78babc796
import Pkg; Pkg.activate(@__DIR__)

# ╔═╡ 6bd11553-ee3a-4ced-a57c-5209359d29f8
using CairoMakie, LinearAlgebra, Colors, PlutoUI, Glob, FileIO, ArnoldiMethod, CacheVariables, Clustering, ProgressLogging, Dates, SparseArrays, Random, Logging, MAT

# ╔═╡ d9e4e1d2-1173-47ad-b5b7-d8548d50d382
html"""<style>
input[type*="range"] {
	width: calc(100% - 4rem);
}
main {
    max-width: 96%;
    margin-left: 0%;
    margin-right: 2% !important;
}
"""

# ╔═╡ 598c7b08-c30f-4b9e-b98a-654b8258c212
@bind Location Select(["Pavia", "PaviaUni"])

# ╔═╡ 329a79b2-c0d4-4014-926c-bfdd043bdd4d
filepath = joinpath(@__DIR__, "MAT Files", "$Location.mat")

# ╔═╡ f180091e-53a4-49af-b1a9-77cef48e08eb
gt_filepath = joinpath(@__DIR__, "GT Files", "$Location.mat")

# ╔═╡ 80da4485-bb98-497c-9ff4-6beb15ba56db
CACHEDIR = joinpath(@__DIR__, "cache_files", "Aerial Datasets")

# ╔═╡ 24b32633-ec87-4007-a9b3-c394dbaa21ac
function cachet(@nospecialize(f), path)
	whenrun, timed_results = cache(path) do
		return now(), @timed f()
	end
	@info "Was run at $whenrun (runtime = $(timed_results.time) seconds)"
	timed_results.value
end

# ╔═╡ 45a93653-61b9-45ea-b595-0c52da56f071
vars = matread(filepath)

# ╔═╡ fa4b50af-b51a-4a08-9ec2-83640bfcad53
vars_gt = matread(gt_filepath)

# ╔═╡ a110405c-ea17-4fda-b636-524bb3121070
loc_dict_keys = Dict(
	"Pavia" => ("pavia", "pavia_gt"),
	"PaviaUni" => ("paviaU", "paviaU_gt")
)

# ╔═╡ e8e42a22-bf2f-4d49-9d9c-6b8fe9f2e77f
data_key, gt_key = loc_dict_keys[Location]

# ╔═╡ 48693eaf-ab34-4c5c-9b4b-92eef5c94b50
data = vars[data_key]

# ╔═╡ e8f6f8f4-d513-415c-b174-fb4f1749ea90
gt_data = vars_gt[gt_key]

# ╔═╡ c5b93b88-35fe-40c6-bdaf-a5c234fe3ee9
gt_labels = sort(unique(gt_data))

# ╔═╡ f1ad7a0d-6461-4d5a-afa6-f94f15331195
bg_indices = findall(gt_data .== 0)

# ╔═╡ 129c5b67-dbab-4924-b85b-f15478b7bfc9
n_clusters = length(unique(gt_data)) - 1

# ╔═╡ d693917b-2b8a-4908-8a38-c54449523af6
with_theme() do
	

	# Create figure
	fig = Figure(; size=(700, 700))
	colors = Makie.Colors.distinguishable_colors(n_clusters+1)

	# Show data
	ax = Axis(fig[1,2]; aspect=DataAspect(), yreversed=true, title="Pavia University")
	hm = heatmap!(ax, permutedims(gt_data); colormap=Makie.Categorical(colors))
	Colorbar(fig[1,1], hm, flipaxis=false)
	colgap!(fig.layout, 1, -200)

	fig
end

# ╔═╡ Cell order:
# ╟─d9e4e1d2-1173-47ad-b5b7-d8548d50d382
# ╠═6750f0bf-429c-407c-a8fd-0ea78babc796
# ╠═6bd11553-ee3a-4ced-a57c-5209359d29f8
# ╠═598c7b08-c30f-4b9e-b98a-654b8258c212
# ╠═329a79b2-c0d4-4014-926c-bfdd043bdd4d
# ╠═f180091e-53a4-49af-b1a9-77cef48e08eb
# ╠═80da4485-bb98-497c-9ff4-6beb15ba56db
# ╠═24b32633-ec87-4007-a9b3-c394dbaa21ac
# ╠═45a93653-61b9-45ea-b595-0c52da56f071
# ╠═fa4b50af-b51a-4a08-9ec2-83640bfcad53
# ╠═a110405c-ea17-4fda-b636-524bb3121070
# ╠═e8e42a22-bf2f-4d49-9d9c-6b8fe9f2e77f
# ╠═48693eaf-ab34-4c5c-9b4b-92eef5c94b50
# ╠═e8f6f8f4-d513-415c-b174-fb4f1749ea90
# ╠═c5b93b88-35fe-40c6-bdaf-a5c234fe3ee9
# ╠═f1ad7a0d-6461-4d5a-afa6-f94f15331195
# ╠═129c5b67-dbab-4924-b85b-f15478b7bfc9
# ╠═d693917b-2b8a-4908-8a38-c54449523af6
