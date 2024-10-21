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

# ╔═╡ e986b3f8-914b-47c8-94aa-0635330d695c
import Pkg; Pkg.activate(@__DIR__)

# ╔═╡ 9531e81f-b0e4-49fc-8946-f6c9ffa04a82
using CairoMakie, LinearAlgebra, Colors, PlutoUI, Glob, FileIO, ArnoldiMethod, CacheVariables, Clustering, ProgressLogging, Dates, SparseArrays, Random, Logging, MAT, ArchGDAL

# ╔═╡ 60918ff2-8f51-11ef-3a74-2def7575fac8
html"""<style>
main {
    max-width: 66%;
    margin-left: 1%;
    margin-right: 2% !important;
}
"""

# ╔═╡ 9abb2611-c08b-4307-8cc2-911302598f12
@bind City Select(["beirut"])

# ╔═╡ 383efc0f-b775-4851-8a55-0ede19c78239
Path = Dict("_T1" => joinpath(@__DIR__, "Onera Satellite Change Detection dataset - Images", City, "imgs_1_rect"), "_T2" => joinpath(@__DIR__, "Onera Satellite Change Detection dataset - Images", City, "imgs_2_rect"))

# ╔═╡ 1606abd2-00f9-4bf6-ae8d-dfc9232e7ef7
FP_1 = Path["_T1"]

# ╔═╡ 8ea9100f-97f7-48b0-92a7-93de81fef45b
FP_2 = Path["_T2"]

# ╔═╡ b277bdcf-b5ad-4b7b-8537-5d323e3f05b6
Images_1 = glob("*.tif", FP_1)

# ╔═╡ 1befe06b-4060-45c5-a47b-70e206246a3e
Images_2 = glob("*.tif", FP_2)

# ╔═╡ d0fbbc1a-9355-4e4f-9ca4-3e823fc993cf
CACHEDIR = joinpath(@__DIR__, "cache_files", "Onera")

# ╔═╡ 613ed99b-e1a3-4a47-af25-c18ed6c074af
function cachet(@nospecialize(f), path)
	whenrun, timed_results = cache(path) do
		return now(), @timed f()
	end
	@info "Was run at $whenrun (runtime = $(timed_results.time) seconds)"
	timed_results.value
end

# ╔═╡ 88559637-55db-4a20-87be-055a1660070e
Array_1 = [ArchGDAL.read(tif_file) do dataset band = ArchGDAL.getband(dataset, 1); data = ArchGDAL.read(band) end for tif_file in Images_1]

# ╔═╡ f327d73f-c6ab-4832-901d-41c8708fb423
Array_2 = [ArchGDAL.read(tif_file) do dataset band = ArchGDAL.getband(dataset, 1); data = ArchGDAL.read(band) end for tif_file in Images_2]

# ╔═╡ 3c98b6d2-f598-49dd-bdeb-000b06a25083
data_1 = cat(Array_1..., dims=3);

# ╔═╡ 4e9b9724-218d-498b-a33c-12d49d5c8887
data_2 = cat(Array_2..., dims=3);

# ╔═╡ 85eab518-fdce-4ed5-b7db-e89d4a3c0239
@bind band PlutoUI.Slider(1:size(data_1, 3), show_value=true)

# ╔═╡ dcf4212b-422f-4d2a-b1f1-2e18416ab147
with_theme() do
	fig = Figure(; size=(600, 500))
	ax_1 = Axis(fig[1, 1], aspect=DataAspect(), yreversed=true)
	ax_2 = Axis(fig[1, 2], aspect=DataAspect(), yreversed=true)
	image!(ax_1, data_1[:, :, band])
	image!(ax_2, data_2[:, :, band])
	fig
end

# ╔═╡ 432bd598-d170-43cc-8519-d0f2758fb083
function affinity(cube; max_nz=10, chunksize=minimum(size(cube)[1:2]),
	func = c -> exp(-2*acos(clamp(c,-1,1))))

	# Verify that chunksize divides the total number of pixels
	mod(prod(size(cube)[1:2]), chunksize) == 0 ||
		error("chunksize must divide the total number of pixels")

	# Compute normalized spectra (so that inner product = cosine of angle)
	X = permutedims(reshape(cube, :, size(cube,3)))
	X = mapslices(normalize, X; dims=1)

	# Find nonzero values (in chunks)
	C_buf = similar(X, size(X,2), chunksize)    # pairwise cosine buffer
	s_buf = Vector{Int}(undef, size(X,2))       # sorting buffer
	nz_list = @withprogress mapreduce(vcat, enumerate(Iterators.partition(1:size(X,2), chunksize))) do (chunk_idx, chunk)
		# Compute cosine angles (for chunk) and store in buffer
		mul!(C_buf, X', view(X, :, chunk))

		# Zero out all but `max_nz` largest values
		nzs = map(chunk, eachcol(C_buf)) do col, c
			idx = partialsortperm!(s_buf, c, 1:max_nz; rev=true)
			collect(idx), fill(col, max_nz), func.(view(c,idx))
		end

		# Log progress and return
		@logprogress chunk_idx/(size(X,2) ÷ chunksize)
		return nzs
	end

	# Form and return sparse array
	rows = reduce(vcat, getindex.(nz_list, 1))
	cols = reduce(vcat, getindex.(nz_list, 2))
	vals = reduce(vcat, getindex.(nz_list, 3))
	return sparse([rows; cols],[cols; rows],[vals; vals])
end

# ╔═╡ 6f1b577a-6042-4fcc-a21b-0b351e829dfd
max_nz = 20

# ╔═╡ 7a59cab0-8ba4-4c6b-8627-696c22fe0218
A_1 = cachet(joinpath(CACHEDIR, "Affinity_$(City)_T1$max_nz.bson")) do
	affinity(data; max_nz)
end

# ╔═╡ 876cc0e8-2880-4591-b212-1eac3d2c0b9a
A_2 = cachet(joinpath(CACHEDIR, "Affinity_$(City)_T2$max_nz.bson")) do
	affinity(data; max_nz)
end

# ╔═╡ 48267064-d704-439f-bd9e-f2f620a902e3
function embedding(A, k; seed=0)
	# Set seed for reproducibility
	Random.seed!(seed)

	# Compute node degrees and form Laplacian
	d = vec(sum(A; dims=2))
	Dsqrinv = sqrt(inv(Diagonal(d)))
	L = Symmetric(I - (Dsqrinv * A) * Dsqrinv)

	# Compute eigenvectors
	decomp, history = partialschur(L; nev=k, which=:SR)
	@info history

	return mapslices(normalize, decomp.Q; dims=2)
end

# ╔═╡ 8b04f5d4-6266-41f6-bacd-a0b091e37f19
n_clusters = 6

# ╔═╡ 2466897c-2361-4ad7-b007-95a14b104bd0
V_1 = embedding(A_1, n_clusters)

# ╔═╡ 6689dd33-a0db-4fa7-9739-4ca079e359dd
V_2 = embedding(A_2, n_clusters)

# ╔═╡ 69646bb2-f708-43dc-b58f-84b1fd8438c2
function batchkmeans(X, k, args...; nruns=100, kwargs...)
	runs = @withprogress map(1:nruns) do idx
		# Run K-means
		Random.seed!(idx)  # set seed for reproducibility
		result = with_logger(NullLogger()) do
			kmeans(X, k, args...; kwargs...)
		end

		# Log progress and return result
		@logprogress idx/nruns
		return result
	end

	# Print how many converged
	nconverged = count(run -> run.converged, runs)
	@info "$nconverged/$nruns runs converged"

	# Return runs sorted best to worst
	return sort(runs; by=run->run.totalcost)
end

# ╔═╡ 66ab6df2-0eb7-4ccd-8577-8412dcf391eb
spec_clusterings_1 = batchkmeans(permutedims(V_1), n_clusters; maxiter=1000)

# ╔═╡ d1c8fb7a-ee78-4601-949f-4c26b668802b
spec_clusterings_2 = batchkmeans(permutedims(V_2), n_clusters; maxiter=1000)

# ╔═╡ da84ec4a-9ad1-432e-b48d-e61f98862e80
costs_1 = [spec_clusterings_1[i].totalcost for i in 1:100]

# ╔═╡ ea2e6763-848f-4aae-8c67-aba2e3ddafb3
costs_2 = [spec_clusterings_2[i].totalcost for i in 1:100]

# ╔═╡ 273b5231-39e4-4225-abd5-b9e6596719ea
clusters_1 = spec_clusterings_1[argmin(costs_1)].assignments

# ╔═╡ d2e56b14-dfe3-44f7-8cad-b6ea6a3960a3
clusters_2 = spec_clusterings_2[argmin(costs_2)].assignments

# ╔═╡ 512f26cb-24de-4387-b6d1-3f1a4650c1e1
mat_1 = reshape(clusters_1, size(data_1, 1), size(data_1, 2))

# ╔═╡ a088fef2-0966-4fb0-90f1-fb006fb8827d
mat_2 = reshape(clusters_2, size(data_1, 1), size(data_1, 2))

# ╔═╡ 7ef2c784-bd0c-4cf4-9d4a-6d4d81d8410c
with_theme() do
	fig = Figure(; size=(900, 600))
	ax_1 = Axis(fig[1, 1], aspect=DataAspect(), yreversed=true)
	ax_2 = Axis(fig[1, 2], aspect=DataAspect(), yreversed=true)
	colors = Makie.Colors.distinguishable_colors(n_clusters)
	hm_1 = heatmap!(ax_1, mat_1; colormap=Makie.Categorical(colors))
	hm_2 = heatmap!(ax_2, mat_2; colormap=Makie.Categorical(colors))
	fig
end

# ╔═╡ Cell order:
# ╟─60918ff2-8f51-11ef-3a74-2def7575fac8
# ╠═e986b3f8-914b-47c8-94aa-0635330d695c
# ╠═9531e81f-b0e4-49fc-8946-f6c9ffa04a82
# ╠═9abb2611-c08b-4307-8cc2-911302598f12
# ╠═383efc0f-b775-4851-8a55-0ede19c78239
# ╠═1606abd2-00f9-4bf6-ae8d-dfc9232e7ef7
# ╠═8ea9100f-97f7-48b0-92a7-93de81fef45b
# ╠═b277bdcf-b5ad-4b7b-8537-5d323e3f05b6
# ╠═1befe06b-4060-45c5-a47b-70e206246a3e
# ╠═d0fbbc1a-9355-4e4f-9ca4-3e823fc993cf
# ╠═613ed99b-e1a3-4a47-af25-c18ed6c074af
# ╠═88559637-55db-4a20-87be-055a1660070e
# ╠═f327d73f-c6ab-4832-901d-41c8708fb423
# ╠═3c98b6d2-f598-49dd-bdeb-000b06a25083
# ╠═4e9b9724-218d-498b-a33c-12d49d5c8887
# ╠═85eab518-fdce-4ed5-b7db-e89d4a3c0239
# ╠═dcf4212b-422f-4d2a-b1f1-2e18416ab147
# ╠═432bd598-d170-43cc-8519-d0f2758fb083
# ╠═6f1b577a-6042-4fcc-a21b-0b351e829dfd
# ╠═7a59cab0-8ba4-4c6b-8627-696c22fe0218
# ╠═876cc0e8-2880-4591-b212-1eac3d2c0b9a
# ╠═48267064-d704-439f-bd9e-f2f620a902e3
# ╠═8b04f5d4-6266-41f6-bacd-a0b091e37f19
# ╠═2466897c-2361-4ad7-b007-95a14b104bd0
# ╠═6689dd33-a0db-4fa7-9739-4ca079e359dd
# ╠═69646bb2-f708-43dc-b58f-84b1fd8438c2
# ╠═66ab6df2-0eb7-4ccd-8577-8412dcf391eb
# ╠═d1c8fb7a-ee78-4601-949f-4c26b668802b
# ╠═da84ec4a-9ad1-432e-b48d-e61f98862e80
# ╠═ea2e6763-848f-4aae-8c67-aba2e3ddafb3
# ╠═273b5231-39e4-4225-abd5-b9e6596719ea
# ╠═d2e56b14-dfe3-44f7-8cad-b6ea6a3960a3
# ╠═512f26cb-24de-4387-b6d1-3f1a4650c1e1
# ╠═a088fef2-0966-4fb0-90f1-fb006fb8827d
# ╠═7ef2c784-bd0c-4cf4-9d4a-6d4d81d8410c
