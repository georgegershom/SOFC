# SENCE Framework Mermaid Diagrams

## 1. SENCE Framework Structure Diagram

```mermaid
graph TB
    subgraph "SENCE Framework"
        S[Social Domain]
        E[Economic Domain]
        N[Natural/Environmental Domain]
        C[Compound Interactions]
        E2[Ecosystem Services]
    end
    
    subgraph "Social Domain Components"
        S1[Human Capital]
        S2[Community Cohesion]
        S3[Access to Services]
        S4[Safety & Security]
        S5[Governance Trust]
    end
    
    subgraph "Economic Domain Components"
        E1[Livelihood Diversity]
        E2[Income Stability]
        E3[Infrastructure Access]
        E4[Employment Opportunities]
        E5[Economic Resilience]
    end
    
    subgraph "Environmental Domain Components"
        N1[Oil Spill Impact]
        N2[Gas Flaring Effects]
        N3[Vegetation Health]
        N4[Water Quality]
        N5[Land Degradation]
        N6[Mangrove Loss]
    end
    
    subgraph "Niger Delta Cities"
        PH[Port Harcourt<br/>CVI: 0.52]
        WA[Warri<br/>CVI: 0.61]
        BO[Bonny<br/>CVI: 0.59]
    end
    
    subgraph "Vulnerability Assessment"
        CVI[Composite Vulnerability Index]
        PCA[Principal Component Analysis]
        NORM[Normalization Process]
        AGG[Aggregation Methods]
    end
    
    %% Domain connections
    S --> S1
    S --> S2
    S --> S3
    S --> S4
    S --> S5
    
    E --> E1
    E --> E2
    E --> E3
    E --> E4
    E --> E5
    
    N --> N1
    N --> N2
    N --> N3
    N --> N4
    N --> N5
    N --> N6
    
    %% Cross-domain interactions
    S -.-> E
    E -.-> N
    N -.-> S
    S -.-> C
    E -.-> C
    N -.-> C
    
    %% City connections
    PH --> CVI
    WA --> CVI
    BO --> CVI
    
    %% Assessment process
    CVI --> PCA
    PCA --> NORM
    NORM --> AGG
    
    %% Styling
    classDef social fill:#e1f5fe,stroke:#01579b,stroke-width:2px
    classDef economic fill:#f3e5f5,stroke:#4a148c,stroke-width:2px
    classDef environmental fill:#e8f5e8,stroke:#1b5e20,stroke-width:2px
    classDef city fill:#fff3e0,stroke:#e65100,stroke-width:3px
    classDef assessment fill:#fce4ec,stroke:#880e4f,stroke-width:2px
    
    class S,S1,S2,S3,S4,S5 social
    class E,E1,E2,E3,E4,E5 economic
    class N,N1,N2,N3,N4,N5,N6 environmental
    class PH,WA,BO city
    class CVI,PCA,NORM,AGG assessment
```

## 2. Vulnerability Assessment Workflow

```mermaid
flowchart TD
    START([Data Collection Phase]) --> SURVEY[Household Surveys]
    START --> GEO[Geospatial Data]
    START --> ENV[Environmental Indices]
    
    SURVEY --> SOC_IND[Social Indicators]
    GEO --> SPAT_IND[Spatial Indicators]
    ENV --> ENV_IND[Environmental Indicators]
    
    SOC_IND --> PREP[Data Preprocessing]
    SPAT_IND --> PREP
    ENV_IND --> PREP
    
    PREP --> CLEAN[Data Cleaning & Validation]
    CLEAN --> NORM[Indicator Normalization]
    
    NORM --> PCA[Principal Component Analysis]
    PCA --> LOAD[Factor Loadings Analysis]
    LOAD --> WEIGHT[Domain Weight Calculation]
    
    WEIGHT --> AGG[Multi-Domain Aggregation]
    AGG --> CVI[Composite Vulnerability Index]
    
    CVI --> RADAR[Radar Chart Visualization]
    CVI --> STAT[Statistical Analysis]
    CVI --> VALID[Model Validation]
    
    RADAR --> INSIGHT[Vulnerability Insights]
    STAT --> INSIGHT
    VALID --> INSIGHT
    
    INSIGHT --> POLICY[Policy Recommendations]
    INSIGHT --> INTERV[Intervention Strategies]
    
    POLICY --> IMPL[Implementation]
    INTERV --> IMPL
    
    IMPL --> MONITOR[Monitoring & Evaluation]
    MONITOR --> FEEDBACK[Feedback Loop]
    FEEDBACK --> START
    
    %% Styling
    classDef process fill:#e3f2fd,stroke:#0277bd,stroke-width:2px
    classDef data fill:#f1f8e9,stroke:#33691e,stroke-width:2px
    classDef analysis fill:#fff8e1,stroke:#f57f17,stroke-width:2px
    classDef output fill:#fce4ec,stroke:#c2185b,stroke-width:2px
    
    class START,END process
    class SURVEY,GEO,ENV,SOC_IND,SPAT_IND,ENV_IND data
    class PREP,CLEAN,NORM,PCA,LOAD,WEIGHT,AGG,CVI analysis
    class RADAR,STAT,VALID,INSIGHT,POLICY,INTERV,IMPL,MONITOR,FEEDBACK output
```

## 3. City-Specific Vulnerability Profiles

```mermaid
graph LR
    subgraph "Port Harcourt Profile"
        PH_ENV[Environmental: 0.45<br/>Moderate Impact]
        PH_ECO[Economic: 0.52<br/>Urban Disparities]
        PH_SOC[Social: 0.48<br/>Marginalization]
        PH_GOV[Governance: 0.41<br/>Moderate Issues]
        PH_INF[Infrastructure: 0.38<br/>Better Access]
    end
    
    subgraph "Warri Profile"
        WA_ENV[Environmental: 0.68<br/>High Pollution]
        WA_ECO[Economic: 0.71<br/>Severe Deprivation]
        WA_SOC[Social: 0.65<br/>Inter-ethnic Conflicts]
        WA_GOV[Governance: 0.58<br/>Challenges]
        WA_INF[Infrastructure: 0.62<br/>Deficits]
    end
    
    subgraph "Bonny Profile"
        BO_ENV[Environmental: 0.89<br/>Extreme Degradation]
        BO_ECO[Economic: 0.76<br/>Mono-dependence]
        BO_SOC[Social: 0.54<br/>Moderate Issues]
        BO_GOV[Governance: 0.47<br/>Moderate]
        BO_INF[Infrastructure: 0.51<br/>Challenges]
    end
    
    %% Connections showing compound effects
    PH_ENV -.-> PH_ECO
    PH_ECO -.-> PH_SOC
    PH_SOC -.-> PH_GOV
    PH_GOV -.-> PH_INF
    
    WA_ENV -.-> WA_ECO
    WA_ECO -.-> WA_SOC
    WA_SOC -.-> WA_GOV
    WA_GOV -.-> WA_INF
    
    BO_ENV -.-> BO_ECO
    BO_ECO -.-> BO_SOC
    BO_SOC -.-> BO_GOV
    BO_GOV -.-> BO_INF
    
    %% Cross-city comparisons
    PH_ENV -.-> WA_ENV
    WA_ENV -.-> BO_ENV
    
    %% Styling
    classDef portHarcourt fill:#e1f5fe,stroke:#01579b,stroke-width:2px
    classDef warri fill:#fff3e0,stroke:#e65100,stroke-width:2px
    classDef bonny fill:#e8f5e8,stroke:#1b5e20,stroke-width:2px
    
    class PH_ENV,PH_ECO,PH_SOC,PH_GOV,PH_INF portHarcourt
    class WA_ENV,WA_ECO,WA_SOC,WA_GOV,WA_INF warri
    class BO_ENV,BO_ECO,BO_SOC,BO_GOV,BO_INF bonny
```

## 4. SENCE Framework Integration Model

```mermaid
graph TB
    subgraph "Input Layer"
        I1[Social Data]
        I2[Economic Data]
        I3[Environmental Data]
        I4[Governance Data]
        I5[Infrastructure Data]
    end
    
    subgraph "Processing Layer"
        P1[Data Normalization]
        P2[Principal Component Analysis]
        P3[Factor Analysis]
        P4[Weight Calculation]
        P5[Composite Index Generation]
    end
    
    subgraph "Integration Layer"
        INT1[Cross-Domain Interactions]
        INT2[Feedback Loops]
        INT3[Multiplicative Effects]
        INT4[System Dynamics]
    end
    
    subgraph "Output Layer"
        O1[Vulnerability Assessment]
        O2[Risk Profiling]
        O3[Policy Recommendations]
        O4[Intervention Strategies]
    end
    
    subgraph "Validation Layer"
        V1[Statistical Validation]
        V2[Model Performance]
        V3[Sensitivity Analysis]
        V4[Uncertainty Quantification]
    end
    
    %% Input to Processing
    I1 --> P1
    I2 --> P1
    I3 --> P1
    I4 --> P1
    I5 --> P1
    
    %% Processing flow
    P1 --> P2
    P2 --> P3
    P3 --> P4
    P4 --> P5
    
    %% Integration
    P5 --> INT1
    INT1 --> INT2
    INT2 --> INT3
    INT3 --> INT4
    
    %% Output generation
    INT4 --> O1
    O1 --> O2
    O2 --> O3
    O3 --> O4
    
    %% Validation
    O1 --> V1
    O2 --> V2
    O3 --> V3
    O4 --> V4
    
    %% Feedback loops
    V1 -.-> P2
    V2 -.-> P3
    V3 -.-> P4
    V4 -.-> P5
    
    %% Styling
    classDef input fill:#e8f5e8,stroke:#2e7d32,stroke-width:2px
    classDef process fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
    classDef integration fill:#fff3e0,stroke:#ef6c00,stroke-width:2px
    classDef output fill:#fce4ec,stroke:#c2185b,stroke-width:2px
    classDef validation fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    
    class I1,I2,I3,I4,I5 input
    class P1,P2,P3,P4,P5 process
    class INT1,INT2,INT3,INT4 integration
    class O1,O2,O3,O4 output
    class V1,V2,V3,V4 validation
```