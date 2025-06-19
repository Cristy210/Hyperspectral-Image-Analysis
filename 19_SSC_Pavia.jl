### A Pluto.jl notebook ###
# v0.20.8

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    #! format: off
    return quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
    #! format: on
end

# ╔═╡ 2c184464-068a-4f96-b910-0da43a386055
import Pkg; Pkg.activate(@__DIR__)

# ╔═╡ c606166d-c02f-4d77-9107-8736889f2ab8
using CairoMakie, LinearAlgebra, Colors, PlutoUI, Glob, FileIO, ArnoldiMethod, CacheVariables, Clustering, ProgressLogging, Dates, SparseArrays, Random, Logging, MAT

# ╔═╡ b4b55c7f-93f7-42ff-b0c3-8ddaac58262b
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

# ╔═╡ d3b8ac62-2f70-4c6f-9d5f-157c432d25d9
@bind Location Select(["Pavia", "PaviaUni",])

# ╔═╡ ab355b03-fc25-421b-9368-6ab268d3a04c
filepath = joinpath(@__DIR__, "MAT Files", "$Location.mat")

# ╔═╡ 348e2ae0-3aaa-40e0-af60-d5aee5c07782
gt_filepath = joinpath(@__DIR__, "GT Files", "$Location.mat")

# ╔═╡ 670c693a-ccd5-41ae-bb40-00994f88699d


# ╔═╡ Cell order:
# ╠═b4b55c7f-93f7-42ff-b0c3-8ddaac58262b
# ╠═2c184464-068a-4f96-b910-0da43a386055
# ╠═c606166d-c02f-4d77-9107-8736889f2ab8
# ╠═d3b8ac62-2f70-4c6f-9d5f-157c432d25d9
# ╠═ab355b03-fc25-421b-9368-6ab268d3a04c
# ╠═348e2ae0-3aaa-40e0-af60-d5aee5c07782
# ╠═670c693a-ccd5-41ae-bb40-00994f88699d
