C     Abaqus Material Subroutine for Concrete Fire Behavior
C     Pillar 3: Numerical Modeling Dataset
C     
      SUBROUTINE UMAT(STRESS,STATEV,DDSDDE,SSE,SPD,SCD,
     1 RPL,DDSDDT,DRPLDE,DRPLDT,STRAN,DSTRAN,TIME,DTIME,TEMP,
     2 DTEMP,PREDEF,DPRED,CMNAME,NDI,NSHR,NTENS,NSTATV,PROPS,
     3 NPROPS,COORDS,DROT,PNEWDT,CELENT,DFGRD0,DFGRD1,NOEL,
     4 NPT,LAYER,KSPT,KSTEP,KINC)
C
      INCLUDE 'ABA_PARAM.INC'
C
      CHARACTER*80 CMNAME
      DIMENSION STRESS(NTENS),STATEV(NSTATV),DDSDDE(NTENS,NTENS),
     1 DDSDDT(NTENS),DRPLDE(NTENS),STRAN(NTENS),DSTRAN(NTENS),
     2 PREDEF(1),DPRED(1),PROPS(NPROPS),COORDS(3),DROT(3,3),
     3 DFGRD0(3,3),DFGRD1(3,3)
C
C     Material properties
      REAL*8 E0, NU, FC0, FT0, ALPHA, BETA, GAMMA, DELTA
      REAL*8 TEMP, DTEMP, E, FC, FT, ALPHA_T
      REAL*8 STRESS_TEMP(NTENS), DDSDDE_TEMP(NTENS,NTENS)
C
C     Initialize material properties
      E0 = PROPS(1)
      NU = PROPS(2)
      FC0 = PROPS(3)
      FT0 = PROPS(4)
      ALPHA = PROPS(5)
      BETA = PROPS(6)
      GAMMA = PROPS(7)
      DELTA = PROPS(8)
C
C     Temperature-dependent properties
      CALL TEMP_PROPERTIES(TEMP, E0, NU, FC0, FT0, ALPHA,
     1                     E, NU, FC, FT, ALPHA_T)
C
C     Calculate elastic modulus matrix
      CALL ELASTIC_MATRIX(E, NU, DDSDDE)
C
C     Calculate thermal strain
      CALL THERMAL_STRAIN(ALPHA_T, DTEMP, STRAN, DSTRAN)
C
C     Calculate stress
      CALL CALCULATE_STRESS(STRESS, DDSDDE, DSTRAN, FC, FT)
C
C     Calculate pore pressure
      CALL PORE_PRESSURE(TEMP, STATEV, RPL)
C
C     Calculate dehydration
      CALL DEHYDRATION(TEMP, DTEMP, STATEV)
C
      RETURN
      END
C
C     Temperature-dependent properties subroutine
      SUBROUTINE TEMP_PROPERTIES(TEMP, E0, NU0, FC0, FT0, ALPHA0,
     1                           E, NU, FC, FT, ALPHA)
C
      REAL*8 TEMP, E0, NU0, FC0, FT0, ALPHA0
      REAL*8 E, NU, FC, FT, ALPHA
C
C     Temperature-dependent reduction factors
      IF (TEMP .LE. 100.0) THEN
          E = E0 * (1.0 - 0.05 * TEMP / 100.0)
          FC = FC0 * (1.0 - 0.05 * TEMP / 100.0)
          FT = FT0 * (1.0 - 0.05 * TEMP / 100.0)
          NU = NU0
          ALPHA = ALPHA0 * (1.0 + 0.1 * TEMP / 100.0)
      ELSE IF (TEMP .LE. 200.0) THEN
          E = E0 * (0.95 - 0.15 * (TEMP - 100.0) / 100.0)
          FC = FC0 * (0.95 - 0.15 * (TEMP - 100.0) / 100.0)
          FT = FT0 * (0.95 - 0.15 * (TEMP - 100.0) / 100.0)
          NU = NU0 + 0.01 * (TEMP - 100.0) / 100.0
          ALPHA = ALPHA0 * (1.1 + 0.1 * (TEMP - 100.0) / 100.0)
      ELSE IF (TEMP .LE. 400.0) THEN
          E = E0 * (0.8 - 0.4 * (TEMP - 200.0) / 200.0)
          FC = FC0 * (0.8 - 0.4 * (TEMP - 200.0) / 200.0)
          FT = FT0 * (0.8 - 0.4 * (TEMP - 200.0) / 200.0)
          NU = NU0 + 0.01 + 0.02 * (TEMP - 200.0) / 200.0
          ALPHA = ALPHA0 * (1.2 + 0.2 * (TEMP - 200.0) / 200.0)
      ELSE IF (TEMP .LE. 600.0) THEN
          E = E0 * (0.4 - 0.3 * (TEMP - 400.0) / 200.0)
          FC = FC0 * (0.4 - 0.3 * (TEMP - 400.0) / 200.0)
          FT = FT0 * (0.4 - 0.3 * (TEMP - 400.0) / 200.0)
          NU = NU0 + 0.03 + 0.02 * (TEMP - 400.0) / 200.0
          ALPHA = ALPHA0 * (1.4 + 0.2 * (TEMP - 400.0) / 200.0)
      ELSE IF (TEMP .LE. 800.0) THEN
          E = E0 * (0.1 - 0.05 * (TEMP - 600.0) / 200.0)
          FC = FC0 * (0.1 - 0.05 * (TEMP - 600.0) / 200.0)
          FT = FT0 * (0.1 - 0.05 * (TEMP - 600.0) / 200.0)
          NU = NU0 + 0.05 + 0.02 * (TEMP - 600.0) / 200.0
          ALPHA = ALPHA0 * (1.6 + 0.2 * (TEMP - 600.0) / 200.0)
      ELSE
          E = E0 * 0.05
          FC = FC0 * 0.05
          FT = FT0 * 0.05
          NU = NU0 + 0.07
          ALPHA = ALPHA0 * 1.8
      END IF
C
      RETURN
      END
C
C     Elastic modulus matrix subroutine
      SUBROUTINE ELASTIC_MATRIX(E, NU, DDSDDE)
C
      REAL*8 E, NU, DDSDDE(6,6)
      REAL*8 C11, C12, C44
C
      C11 = E * (1.0 - NU) / ((1.0 + NU) * (1.0 - 2.0 * NU))
      C12 = E * NU / ((1.0 + NU) * (1.0 - 2.0 * NU))
      C44 = E / (2.0 * (1.0 + NU))
C
      DDSDDE = 0.0
      DDSDDE(1,1) = C11
      DDSDDE(2,2) = C11
      DDSDDE(3,3) = C11
      DDSDDE(1,2) = C12
      DDSDDE(2,1) = C12
      DDSDDE(1,3) = C12
      DDSDDE(3,1) = C12
      DDSDDE(2,3) = C12
      DDSDDE(3,2) = C12
      DDSDDE(4,4) = C44
      DDSDDE(5,5) = C44
      DDSDDE(6,6) = C44
C
      RETURN
      END
C
C     Thermal strain subroutine
      SUBROUTINE THERMAL_STRAIN(ALPHA, DTEMP, STRAN, DSTRAN)
C
      REAL*8 ALPHA, DTEMP, STRAN(6), DSTRAN(6)
C
      DSTRAN(1) = DSTRAN(1) - ALPHA * DTEMP
      DSTRAN(2) = DSTRAN(2) - ALPHA * DTEMP
      DSTRAN(3) = DSTRAN(3) - ALPHA * DTEMP
C
      RETURN
      END
C
C     Calculate stress subroutine
      SUBROUTINE CALCULATE_STRESS(STRESS, DDSDDE, DSTRAN, FC, FT)
C
      REAL*8 STRESS(6), DDSDDE(6,6), DSTRAN(6), FC, FT
      REAL*8 STRESS_TEMP(6)
      INTEGER I, J
C
      DO I = 1, 6
          STRESS_TEMP(I) = 0.0
          DO J = 1, 6
              STRESS_TEMP(I) = STRESS_TEMP(I) + DDSDDE(I,J) * DSTRAN(J)
          END DO
      END DO
C
      STRESS = STRESS_TEMP
C
      RETURN
      END
C
C     Pore pressure subroutine
      SUBROUTINE PORE_PRESSURE(TEMP, STATEV, RPL)
C
      REAL*8 TEMP, STATEV(10), RPL
      REAL*8 PORE_PRESSURE, VAPOR_PRESSURE
C
      IF (TEMP .GT. 100.0) THEN
          VAPOR_PRESSURE = 101325.0 * EXP(8.07131 - 1730.63 / (TEMP + 233.426))
          PORE_PRESSURE = VAPOR_PRESSURE * (TEMP - 100.0) / 100.0
          STATEV(1) = PORE_PRESSURE
          RPL = PORE_PRESSURE
      ELSE
          STATEV(1) = 101325.0
          RPL = 0.0
      END IF
C
      RETURN
      END
C
C     Dehydration subroutine
      SUBROUTINE DEHYDRATION(TEMP, DTEMP, STATEV)
C
      REAL*8 TEMP, DTEMP, STATEV(10)
      REAL*8 MOISTURE_CONTENT, DEHYDRATION_RATE
C
      MOISTURE_CONTENT = STATEV(2)
      DEHYDRATION_RATE = 0.0
C
      IF (TEMP .GT. 105.0 .AND. MOISTURE_CONTENT .GT. 0.0) THEN
          DEHYDRATION_RATE = 0.001 * EXP(-45000.0 / (8.314 * TEMP))
          MOISTURE_CONTENT = MOISTURE_CONTENT - DEHYDRATION_RATE * DTEMP
          IF (MOISTURE_CONTENT .LT. 0.0) MOISTURE_CONTENT = 0.0
      END IF
C
      STATEV(2) = MOISTURE_CONTENT
C
      RETURN
      END