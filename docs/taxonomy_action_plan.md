# Taxonomy Action Plan Based On Real Budget Bases

## What We Measured

The seed was generated from four real bases:

- `taxonomy_inputs/base_teste3.xlsx` - description in column `C`
- `taxonomy_inputs/base_teste4.xlsx` - description in column `C`
- `taxonomy_inputs/base_dados.xlsx` - description in column `G`
- `taxonomy_inputs/contraprova.xlsx` - description in column `G`

Totals from the current extractor:

- `13069` descriptions read
- `31` families currently detected
- `18` subfamilies currently detected
- `13` materials currently detected
- `5964` descriptions still unclassified

That means the current taxonomy is useful, but still far from covering the real domain.

## Clear Structural Problems Found

### 1. High unclassified volume

Many descriptions are not material/item descriptions at all. They include:

- mobilization
- licenses and fees
- site setup
- transport moments
- labor roles
- demolition and logistics phrases

This means we need a service/administrative branch in the taxonomy instead of treating everything as item matching.

### 2. False family activations by generic words

Examples seen in the seed:

- `terminal` is firing for texts like `circuitos terminais`
- `painel` is firing for `painel` as construction element, not electrical panel
- `piso` is being activated in contexts that are actually demolition or roofing

This means the current alias list is too flat. We need context-sensitive activation, not just string presence.

### 3. Technical regex noise

The frequency report shows values like:

- `06/2022`
- `09/2023`

inside the `polegada` bucket.

So the current inch parser is accidentally reading dates as technical dimensions.

### 4. Family coverage gaps in important domains

The current top families show good traction in:

- concrete
- cables
- tubes
- painting
- conduits
- fittings

But we still need stronger structured coverage for domains like:

- electrical protection and switching
- industrial piping accessories
- civil temporary works
- HVAC / insulation
- fire-fighting assemblies
- sanitary ware and plumbing sets

## Structured Plan

## Phase 1 - Clean extraction layer

Goal: make the taxonomy seed trustworthy before using it to drive matching.

Tasks:

- ignore obvious header rows such as `DESCRIÇÃO`
- create a `service_or_admin` branch for non-item scopes
- prevent dates from entering inch and diameter extractors
- split quantity/unit/date noise from technical attributes

## Phase 2 - Family model by domain

Goal: move from a flat family list to a domain-aware taxonomy.

Initial macro-domains:

- `civil`
- `hidrossanitario`
- `incendio`
- `eletrica`
- `tubulacao_industrial`
- `hvac`
- `canteiro_e_administracao`

Each macro-domain should contain:

- families
- subfamilies
- negative competitors
- required attributes
- preferred synonyms

## Phase 3 - Context-sensitive family activation

Goal: stop generic words from hijacking the ranking.

Examples:

- `terminal` should not activate just because of `circuitos terminais`
- `painel` should distinguish electrical panel from sandwich facade panel
- `te` should outrank `cotovelo` when the search explicitly asks for `te`
- `flange cego` should not collapse into `junta para flange`

This should become a proper rule layer, not scattered patches.

## Phase 4 - Candidate retrieval constrained by plausible family

Goal: improve the candidate pool before the LLM is called.

Planned behavior:

- lexical retrieval stays broad
- family inference narrows the plausible search band
- reranking happens mainly inside that band
- the LLM only sees plausible candidates from the right family

This is the biggest structural gain for the low-score cases we have been seeing.

## Phase 5 - Regression suite from real lines

Goal: stop fixing one case and breaking another.

We should maintain a benchmark set with:

- source base
- target description
- expected family
- acceptable candidate rows
- acceptable fallback behavior when the exact item does not exist

The first benchmark should include the lines we already audited from `Teste 3`.

## What This Enables Next

Once the taxonomy seed is stable, we can:

1. convert it into code-native taxonomy files
2. refactor `extrair_atributos_tecnicos()` and family inference to use that taxonomy
3. rerun the benchmark lines automatically before each push

This is the path to stop reacting case by case and start improving the motor systematically.
