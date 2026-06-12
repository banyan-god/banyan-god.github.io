---
title: "4× RTX PRO 6000 Blackwell on Water, and the One Card That Wouldn't Behave"
date: 2026-06-12
description: "Converting four RTX PRO 6000 Blackwell cards to waterblocks, finding a VRM choke loose on the workbench, and getting back to 41k tok/s."
author: "Sabareesh"
draft: false
summary: "Converting four RTX PRO 6000 Blackwell cards to waterblocks, finding a VRM choke loose on the workbench, and getting back to 41k tok/s."
cover:
  image: "/blog-images/wb-rig-4gpu.jpg"
  alt: "Four waterblocked RTX PRO 6000 Blackwell GPUs on an open test bench"
  relative: false
images:
  - "/blog-images/wb-rig-4gpu.jpg"
tags: ["hardware", "gpu", "blackwell", "watercooling", "rtx-pro-6000", "vllm", "debugging"]
categories:
  - Hardware
  - GPU
---

Four RTX PRO 6000 Blackwell cards in one chassis at 600 W each is 2.4 kW of heat to evict. Air coolers can do it, but the fans are loud and the cards stack their exhaust into each other. So we converted all four to waterblocks. Most of the build went fine. One didn't — and the reason was sitting on the workbench.

This post is the short version: what we did, what broke, how we found it, and where we landed.

## The rig

![Four waterblocked RTX PRO 6000 Blackwells in the open bench](/blog-images/wb-rig-4gpu.jpg)

- 4× RTX PRO 6000 Blackwell Workstation (GB202, 96 GB GDDR7, 600 W)
- Threadripper Pro 7995WX on WRX90
- 4× Bykski waterblocks (full-cover, GPU + VRM + memory front-side)
- Custom loop: single distro/reservoir, two pumps, distilled water, two Alphacool NexXxoS XT45 Full Copper 1260 mm Super Nova radiators (9× 140 mm fans each), four GPUs plumbed in parallel
- 2× PSUs to feed the ~2.4 kW sustained draw; AC circuit got upgraded mid-build after an earlier all-cards-down event under load

The waterblocks themselves are straightforward: pull the stock cooler, clean the die, fresh paste on the GPU, thermal pads on memory and VRMs, torque the block down in a star pattern. The catch on these cards is the backplate — the memory packages on the back also need cooling, which means either pads against the case panel or small finned heatsinks glued on with thermal adhesive. I went with HOAOH 2.0 W/m·K tape on most spots and GENNEL G109 thermal adhesive where I needed something that wouldn't migrate.

## The card that wouldn't behave

Three cards came up clean. The fourth — GPU 1 on this rig — would idle fine, then fall off the bus under load. The dmesg signature was always the same:

```
NVRM: Xid (PCI:0000:02:00): 79, pid='<unknown>', GPU has fallen off the bus.
NVRM: Xid (PCI:0000:02:00): 154, Node Reboot Required
```

Xid 79 by itself is a generic "GPU stopped responding" — it can be driver, PCIe link, power, or the card. The companion 154 plus the PCIe AER logs showed a DPC containment event: the root port killed the link because the card stopped acknowledging transactions. That narrowed it to the card or its power delivery, not software.

The painful part is that everything else looked normal. The card enumerated. It loaded the driver. It ran short workloads. It only failed after the VRMs had been driving real current for a while.

The temptation here is to chase software: try a different driver, a different vLLM build, swap CUDA versions, blame torch.compile. I tried some of that. None of it changed anything. The next step was to stop guessing and look at the card.

## Pulling the block

![PCB with the waterblock removed, one VRM pad empty](/blog-images/wb-bare-pcb.jpg)

This is the back side of the GPU with the block off. The big metal lid in the middle is the GB202 IHS. The black ring around it is the VRM — each of those small black squares marked **85N** is a power inductor (a choke). They sit between the VRM MOSFETs and the GPU core, smoothing the switched current that feeds the die.

A 600 W card has a lot of these chokes for a reason. They share the load. Lose one and the rest pick up its share, but the regulator's feedback loop gets unhappy and the current waveform gets noisy.

If you look at the upper-right cluster of chokes, one pad is empty. There are two bare solder lands with nothing on them.

![Empty pad close-up, with the missing part on the cloth above](/blog-images/wb-empty-pad.jpg)

The two shiny rectangles are the landing pads. The component that should be bridging them is gone.

It was on the bench.

## The part

![The detached choke on a beige cloth](/blog-images/wb-choke-found.jpg)

![Same part, 85N marking visible](/blog-images/wb-choke-85n.jpg)

About 3 mm on a side, marked **85N**, identical to the 23 still on the board. At some point during the waterblock conversion — most likely while peeling the stock thermal pad off the VRM area — the choke came off with the pad and ended up on the mat. It's small enough that it didn't get noticed during reassembly.

Now the failure mode makes sense. Idle and light loads: the remaining chokes carry the current without complaint. Sustained inference at 600 W: ripple climbs, one of the GPU's internal rails dips out of spec, and the card aborts the link rather than corrupt data. Hence Xid 79 only under real load, and only on this one card.

## Putting it back

Resoldering a power inductor onto a multi-layer GPU PCB is not glamorous work but it isn't exotic either. I did it with a **$40 SmartFix soldering kit from Amazon** — not a $500 rework station. Flux the pads, tin them lightly, place the part, reflow with the surrounding area shielded. The pads on these inductors are big and flat, which actually makes them easier to land than the fine-pitch stuff next to them. Visual check under magnification, continuity check across the part, reinstall the waterblock with fresh paste and pads, back in the loop.

If you're hesitating because you don't own pro rework gear: you don't need it for a part this size. A cheap kit, patience, and a steady hand are enough.

Powered on. `nvidia-smi` showed all four cards. Ran the standard stress suite on the repaired card alone:

```
Inference (vLLM):   10,283 tok/s sustained over 26 rounds
Training (PyTorch): 5,389 tok/s, ~213 TFLOPS, peak 54 °C, 609 W
Xid events:         none
```

Then all four together, same suite, started simultaneously:

```
                inference        training         peak temp    power
GPU 0           10,393 tok/s     210.5 TFLOPS     57 °C        612 W
GPU 1 (fixed)   10,234 tok/s     212.5 TFLOPS     54 °C        609 W
GPU 2           10,311 tok/s     208.3 TFLOPS     58 °C        608 W
GPU 3           10,143 tok/s     206.2 TFLOPS     58 °C        607 W
--------------------------------------------------------------------
Aggregate       41,081 tok/s     837.6 TFLOPS     58 °C       2.44 kW
Xid events: 0
```

The repaired card runs the coolest of the four — fresh paste and pads. The other three are within 4% of each other on inference and within 3% on training, which is about as tight as fleet matching gets on stock silicon. The water loop holds every card at full boost indefinitely; on air, the same workload throttled the cards to the mid-80s °C and clocked them down.

## What I'd do differently

A few small things that would have caught this on day one:

- **Photograph the back of the card before you start.** Side-by-side with the post-reassembly shot, a missing 3 mm component is obvious. Without the reference, it's invisible.
- **Count parts after pad removal.** The thermal pads on stock coolers are sticky enough to pull small SMD components off if they're already poorly bonded from the factory. Anything that comes off with the pad should be found before the block goes back on.
- **Don't chase software when the failure signature points at the card.** Xid 79 with a DPC containment event under load and only under load is a hardware signal. I spent a few hours on driver and inference-engine theories I should have skipped.

## What the rig is doing now

The four cards run a DP=4 vLLM deployment of Qwen3.6-27B — one independent instance per GPU, an nginx load balancer in front, exposed through a Cloudflare tunnel. At 1,024 concurrent in-flight requests with 5,120-token outputs, it does just over **8,000 output tokens per second** sustained, balanced to within 0.7% across the four endpoints, with KV cache at 99.6% and every card pulling its full 600 W. The synthetic stress hits ~41k tok/s because batches are shorter and prefix overhead is hidden; the production-shaped workload at 5k-token generations runs the cards just as hard but spends more cycles on KV. The waterblocks are why that's a 24/7 workload and not a 10-minute demo.

The 85N inductor is back where it belongs.
